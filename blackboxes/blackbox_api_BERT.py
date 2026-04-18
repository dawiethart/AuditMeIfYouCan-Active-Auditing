# blackbox_api.py
from __future__ import annotations
from typing import List, Optional, Dict, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm




def _to_probs(logits: torch.Tensor, num_labels: int) -> torch.Tensor:
    if num_labels == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=-1)[..., 1]


class _TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, groups: List[str]):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.groups = groups

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        return {
            "text": self.texts[i],
            "label": self.labels[i],
            "group": self.groups[i],
        }


class BlackBoxAPI:
    """
    Black-box scorer backed by a Transformer classifier (e.g., hateBERT).
    - predict_scores(texts) -> P(y=1|x) for each text (no hard decisions).
    - train(...) optionally fine-tunes the model and injects bias by flipping labels
      with group-conditional probabilities.
    """

    def __init__(
        self,
        model_name_or_path: str = "GroNLP/hateBERT",
        tokenizer_name_or_path: Optional[str] = None,
        max_length: int = 256,
        default_batch_size: int = 32,
        use_fp16_on_cuda: bool = False,
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or model_name_or_path
        self.max_length = max_length
        self.default_batch_size = default_batch_size

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)

        # Use a single-logit head for clean probability shaping
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=1
        )
        self.model = self.model.to(self.device)
        self.mixed_precision = (self.device.type == "cuda" and use_fp16_on_cuda)

        self.model.eval()
        self.num_labels = 1

    # ----------------------- Training with group bias -----------------------
    def train(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        groups: List[str],
        *,
        flip_probs: Optional[Dict[str, float]] = None,
        epochs: int = 2,
        batch_size: Optional[int] = None,
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.06,
        seed: int = 42,
        shuffle: bool = True,
        grad_accum: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Fine-tune the black-box on provided data and inject **bias** by flipping labels
        with per-group probabilities in `flip_probs`.

        Args
        ----
        flip_probs: dict like {"black": 0.9, "white": 0.1}
            Probability of flipping the gold label for each group. Omitted groups
            default to 0.0 (no flip).
        """
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels).astype(np.int64)
        batch_size = batch_size or self.default_batch_size
        flip_probs = flip_probs or {"black": 0.9, "white": 0.1}  # <- matches “we flipped black and white”

        # --- Inject bias by stochastic label flipping per group ---------------
        flipped = labels.copy()
        for g_name, p in flip_probs.items():
            if p <= 0: continue
            idx = np.array([i for i, g in enumerate(groups) if str(g).lower() == g_name.lower()])
            if idx.size:
                flip_mask = rng.random(idx.size) < p
                flipped[idx[flip_mask]] = 1 - flipped[idx[flip_mask]]

        ds = _TextDataset(texts, flipped.astype(np.float32), groups)

        def collate(batch):
            toks = self.tokenizer(
                [b["text"] for b in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            y = torch.tensor([b["label"] for b in batch], dtype=torch.float32).unsqueeze(-1)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            y = y.to(self.device)
            return toks, y

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

        # Optimizer / scheduler
        self.model.train()
        # AdamW from HF defaults
        optim = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = epochs * len(dl) // max(1, grad_accum)
        warmup_steps = int(warmup_ratio * total_steps)
        sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        step = 0
        for epoch in range(epochs):
            progress_bar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            epoch_loss = 0.0

            for toks, y in dl:
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    out = self.model(**toks)
                    logits = out.logits  # [B,1]
                    loss = loss_fn(logits, y)

                scaler.scale(loss / grad_accum).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()

                step += 1
                progress_bar.set_postfix({"loss": f"{epoch_loss / (step+1):.4f}"})
            progress_bar.close()
        self.model.eval()
    # ------------------------------------------------------------------------

    @torch.inference_mode()
    def predict_scores(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not isinstance(texts, (list, tuple)):
            raise TypeError("`texts` must be a list/tuple of strings.")

        bs = batch_size or self.default_batch_size
        probs_all = []
        for i in tqdm(range(0, len(texts), bs)):
            chunk = texts[i : i + bs]
            enc = self.tokenizer(
                list(chunk),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            probs = _to_probs(outputs.logits, self.num_labels).detach().float().cpu()
            probs_all.append(probs)
        return torch.cat(probs_all, dim=0).numpy()

    # Keep your external API adapters if needed
    def predict_scores_amazon(self, texts: List[str]) -> np.ndarray:
        return scores_amazon(texts)

    def predict_scores_openai(self, texts: List[str]) -> np.ndarray:
        return scores_openai(texts)
