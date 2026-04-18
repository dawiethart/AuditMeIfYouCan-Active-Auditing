# surrogate_model.py

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics.functional import auroc
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from torch.nn import MarginRankingLoss
from transformers import DataCollatorWithPadding

#def load_lora_bert_surrogate(model_name="GroNLP/hateBERT", num_labels=1):
def load_lora_bert_surrogate(model_name="GroNLP/hateBERT", num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # ---- enable hidden states (needed for diversity / embeddings) ----
    base_model.config.output_hidden_states = True
    base_model.config.return_dict = True

    for param in base_model.parameters():
        param.requires_grad = False
    for name, param in base_model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    lora_modules = ["query", "value"] if "bert" in model_name.lower() \
               else ["q_lin", "k_lin", "v_lin", "out_lin"]

    lora_config = LoraConfig(
        target_modules=lora_modules,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base_model, lora_config)

    return tokenizer, model


def train_surrogate(model, tokenizer, df_S_mapped, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./tmp_cerm",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        disable_tqdm=True,
        learning_rate=5e-4,
        warmup_ratio=0.2,
        max_grad_norm=0.5,
        remove_unused_columns=False,
        weight_decay=0.01,
        gradient_accumulation_steps=2
    )

    class CustomDataCollator(DataCollatorWithPadding):
        def __call__(self, features):
            # Keys you want to keep but NOT send to the model
            preserve_keys = ["id", "bb_score", "text"]

            # Keys HuggingFace models accept
            model_keys = ["input_ids", "attention_mask", "labels"]

            # Store extra metadata
            preserved = {k: [f[k] for f in features] if k in features[0] else None
                        for k in preserve_keys}

            # Remove all keys not intended for the model
            for f in features:
                for key in list(f.keys()):
                    if key not in model_keys:
                        f.pop(key, None)

            # Let HF handle input_ids, attention_mask, labels
            batch = super().__call__(features)

            # Reattach metadata
            if preserved["id"] is not None:
                batch["id"] = torch.tensor([int(v) for v in preserved["id"]])

            if preserved["bb_score"] is not None:
                batch["bb_score"] = torch.tensor(preserved["bb_score"], dtype=torch.float32)

            if preserved["text"] is not None:
                batch["text"] = preserved["text"]

            return batch


   
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
           
            bb = inputs.pop("bb_score").float().view(-1)
            labels = inputs.pop("labels").float().view(-1)

            inputs.pop("id", None)
            inputs.pop("text", None)

            outputs = model(**inputs)
            logits = outputs.logits
            if logits.dim()==2 and logits.size(-1)==1:
                logits = logits.squeeze(-1)

            # ensure same device as logits (robust to DP/DDP)
            dev = logits.device
            bb = bb.to(dev); labels = labels.to(dev)

            probs = torch.sigmoid(logits)
            bb_clamped = bb.clamp(1e-6, 1-1e-6)
            mse = torch.nn.functional.mse_loss(probs, bb_clamped)

            # Use all pairs for stable ranking loss (fixed issue 2)
            B = logits.size(0)
            if B >= 2:
                # Create all pairwise comparisons
                i_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand(B, B)
                j_idx = torch.arange(B, device=logits.device).unsqueeze(0).expand(B, B)
                
                # Only keep upper triangular (i < j) to avoid redundant pairs
                mask = i_idx < j_idx
                i = i_idx[mask]
                j = j_idx[mask]
                
                s1, s2 = logits[i], logits[j]
                t = torch.sign(bb[i] - bb[j])
                t[t == 0] = 1.0
                
                # Removed clamping (fixed issue 3) - let gradients flow naturally
                s1_norm = torch.sigmoid(s1)
                s2_norm = torch.sigmoid(s2)
                rank_loss = MarginRankingLoss(margin=0.1)(s1_norm, s2_norm, t)
            else:
                rank_loss = torch.tensor(0.0, device=logits.device)
       
            loss = 0.2*mse + 0.8*rank_loss
            # Removed gradient clipping hook (fixed issue 1) - rely on trainer's max_grad_norm
             
            if self.state.global_step % 10 == 0:
                print(f"Step {self.state.global_step:3d} | MSE={mse.item():.4f} | Rank={rank_loss.item():.4f} | "
                    f"Loss={loss.item():.4f} | Logits:[{logits.min().item():.2f},{logits.max().item():.2f}] | "
                    f"Probs:[{probs.min().item():.3f},{probs.max().item():.3f}]")
    
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=df_S_mapped,
        data_collator=CustomDataCollator(tokenizer=tokenizer, padding=True),  # <-- pass tokenizer object
    )
    trainer.train()
    return model



@torch.no_grad()
def compute_group_auc_difference(model, inputs, dataset, group1=0, group2=1, batch_size=128):
    device = next(model.parameters()).device
    model.eval()

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)
    groups = torch.tensor(dataset["group"].values).to(device)

    logits = []
    for i in range(0, len(input_ids), batch_size):
        batch_logits = model(
            input_ids=input_ids[i : i + batch_size],
            attention_mask=attention_mask[i : i + batch_size],
        ).logits.view(-1)
        logits.append(batch_logits)

    logits = torch.cat(logits)
    probs = torch.sigmoid(logits)

    mask1 = groups == group1
    mask2 = groups == group2
    auc1 = auroc(probs[mask1], labels[mask1].long(), task="binary").item() if mask1.any() else 0.0
    auc2 = auroc(probs[mask2], labels[mask2].long(), task="binary").item() if mask2.any() else 0.0

    """
    def roc_auc(data):
        auc1 = auroc(torch.tensor(data.probs[data.groups == group1].values), 
                     torch.tensor(data.labels[data.groups == group1].values).long(), task="binary")
        auc2 = auroc(torch.tensor(data.probs[data.groups == group2].values),
                     torch.tensor(data.labels[data.groups == group2].values).long(), task="binary")

        return abs(auc1 - auc2)

    contribution = []
    data = pd.DataFrame([probs, labels, groups], index=['probs', 'labels', 'groups']).transpose()
    convert_dict = {'probs': np.float64, 'labels': int, 'groups':int}
    data = data.astype(convert_dict)

    for i in range(len(probs)):
        data_min = pd.concat([data.iloc[0:i], data.iloc[i+1:]])
        con_bootstrap = []
        for j in range(100): # Bootstrapping
            data_min_boot = data_min.sample(n = len(data_min)-1, replace = True)
            roc_auc_bootsrapping = roc_auc(data_min_boot)
            con_bootstrap.append(float(roc_auc_bootsrapping))
        contribution.append(np.array(con_bootstrap).mean())
    """
    return torch.tensor(abs(auc1 - auc2), device=device), probs.cpu()


import torch.nn.functional as F

@torch.no_grad()
def predict_with_model(
    texts,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 128,
    layer: int = -1,          # which hidden state layer to use (-1 = last)
    normalize: bool = True,   # L2 normalize embeddings (recommended)
):
    """
    Returns:
      probs: (n,) numpy float
      embs:  (n, hidden_dim) numpy float  (CLS embeddings from chosen layer)

    Notes:
      - Requires model to support output_hidden_states=True.
      - Works with PEFT LoRA models (PeftModel) if base model supports it.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**inputs, output_hidden_states=True, return_dict=True)

        logits = out.logits
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        probs = torch.sigmoid(logits)  # (b,)

        hs = out.hidden_states
        if hs is None:
            raise RuntimeError(
                "hidden_states is None. Make sure output_hidden_states=True and/or "
                "base_model.config.output_hidden_states=True."
            )

        # CLS embedding from chosen layer
        emb = hs[layer][:, 0, :]  # (b, hidden_dim)

        if normalize:
            emb = F.normalize(emb, p=2, dim=1)

        all_probs.append(probs.detach().cpu())
        all_embs.append(emb.detach().cpu())

    probs_np = torch.cat(all_probs, dim=0).numpy().astype(float)
    embs_np = torch.cat(all_embs, dim=0).numpy().astype(float)
    return probs_np, embs_np