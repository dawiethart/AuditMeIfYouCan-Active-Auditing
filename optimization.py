# optimization.py

import gc
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import torch

from datasets import Value
from torch import amp
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torchmetrics.functional import auroc
from sklearn.metrics import roc_auc_score

from cooper import (
    CMPState,
    ConstrainedMinimizationProblem,
    Constraint,
    ConstraintState,
    ConstraintType,
)
from cooper.multipliers import DenseMultiplier
from cooper.optim import SimultaneousOptimizer


# =========================
#   METRIC / INFERENCE UTILS
# =========================

@torch.no_grad()
def compute_group_auc_difference_from_scores(
    scores,               # 1D array-like of probs in [0,1]
    dataset: pd.DataFrame,              # pandas DataFrame with 'true_label' and 'group'
    group1=1,
    group2=0,
    device: str = "cpu",
) -> Tuple[float, torch.Tensor]:
    """
    ΔAUC = AUC(group2) − AUC(group1) from probability scores.

    This is the *true* ROC-AUC-based ΔAUC, used only for debugging now.
    The bounds we use for sampling are based on the pairwise surrogate ΔAUC.
    """
    probs = torch.as_tensor(np.asarray(scores, dtype=np.float32), device=device).view(-1)

    # labels → int64 {0,1}
    if not np.issubdtype(dataset["true_label"].dtype, np.number):
        labels_np = pd.to_numeric(dataset["true_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        labels_np = dataset["true_label"].to_numpy(dtype=np.int64)
    labels = torch.as_tensor(labels_np, device=device, dtype=torch.long).view(-1)

    # groups → int codes even if strings
    gcol = dataset["group"]
    if not np.issubdtype(gcol.dtype, np.number):
        uniq, inv = np.unique(gcol.astype(str).to_numpy(), return_inverse=True)
        groups_np = inv.astype(np.int64)
    else:
        uniq = None
        groups_np = gcol.to_numpy(dtype=np.int64)
    groups = torch.as_tensor(groups_np, device=device, dtype=torch.long).view(-1)

    # length check
    n = probs.numel()
    if not (n == labels.numel() == groups.numel()):
        raise ValueError(f"Length mismatch: probs={probs.shape}, labels={labels.shape}, groups={groups.shape}")

    def _to_code(g):
        if uniq is None:
            return int(g)
        if isinstance(g, (str, np.str_)):
            code = np.where(uniq == g)[0]
            if len(code) == 0:
                raise ValueError(f"Group '{g}' not found in dataset['group'].")
            return int(code[0])
        return int(g)

    if not np.issubdtype(dataset["group"].dtype, np.number):
        group1 = _to_code(group1)
        group2 = _to_code(group2)

    mask1 = (groups == int(group1))
    mask2 = (groups == int(group2))

    def safe_auroc(p, y):
        return auroc(p, y, task="binary") if y.unique().numel() >= 2 else torch.tensor(float("nan"), device=device)

    auc1 = safe_auroc(probs[mask1], labels[mask1])
    auc2 = safe_auroc(probs[mask2], labels[mask2])
    return (auc2 - auc1).item(), probs.cpu()


@torch.no_grad()
def predict_scores_batched(
    h,
    inputs_D: Dict[str, torch.Tensor],
    batch_size: int = 256,
    device: str = "cuda",
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Batched inference for HF-style dict inputs.
    - inputs_D: dict with 'input_ids', 'attention_mask', optional 'token_type_ids'
    - Returns: scores [N] in CPU float32.
    """
    keys = [k for k in ["input_ids", "attention_mask"] if k in inputs_D]
    if "token_type_ids" in inputs_D:
        keys.append("token_type_ids")

    # ensure CPU base tensors
    N = None
    tensors_cpu: Dict[str, torch.Tensor] = {}
    for k in keys:
        arr = inputs_D[k]
        if not isinstance(arr, torch.Tensor):
            arr = torch.as_tensor(arr)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        tensors_cpu[k] = arr.cpu()
        N = arr.size(0) if N is None else N
    assert all(tensors_cpu[k].size(0) == N for k in keys), "Input dict lengths mismatch."

    h = h.to(device)
    h.eval()

    scores = []
    amp_ctx = amp.autocast if (device.startswith("cuda") and use_amp) else torch.cpu.amp.autocast
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = {k: v[start:end].to(device, non_blocking=True) for k, v in tensors_cpu.items()}
        with amp_ctx(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            outputs = h(**batch)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                batch_scores = torch.sigmoid(logits.squeeze(-1))
            else:
                batch_scores = torch.softmax(logits, dim=-1)[:, 1]
        scores.append(batch_scores.detach().float().cpu())
        del batch, outputs, logits, batch_scores
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    scores = torch.cat(scores, dim=0)  # [N]
    return scores


# =========================
#   DATA COLLATOR
# =========================

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        preserved = {
            key: [f[key] for f in features] for key in ["id", "group", "text"] if key in features[0]
        }
        for f in features:
            for key in preserved:
                f.pop(key, None)
        batch = super().__call__(features)

        for key, values in preserved.items():
            if key == "id":
                try:
                    values = [int(v) for v in values]
                except ValueError:
                    unique = {v: i for i, v in enumerate(sorted(set(values)))}
                    values = [unique[v] for v in values]
                batch[key] = torch.tensor(values)
            elif key == "group":
                try:
                    batch[key] = torch.tensor(values)
                except Exception:
                    # if group is non-numeric and we don't need it as tensor, leave as-is
                    pass
            else:
                batch[key] = values
        return batch


# =========================
#   C-ERM PROBLEM (PAIRWISE)
# =========================

class PairwiseCERMProblem(ConstrainedMinimizationProblem):
    """
    Pairwise C-ERM for maximizing (or minimizing) a surrogate ΔAUC under
    per-example constraints on T. This version operates on *probabilities*
    (sigmoid(logits)) internally to avoid extreme-logit scaling effects.
    """
    def __init__(self, model, inputs_T, constraint_pred, lambda_penalty, maximize, device):
        super().__init__()
        self.model = model
        self.inputs_T = inputs_T              # keep on CPU; we'll slice to GPU
        self.constraint_pred = constraint_pred  # CPU tensor of size |T|
        self.lambda_penalty = lambda_penalty
        self.maximize = maximize
        self.device = device

        num_constraints = int(self.inputs_T["input_ids"].size(0))
        self.multiplier = DenseMultiplier(num_constraints=num_constraints, device=device)
        self.constraint = Constraint(
            multiplier=self.multiplier,
            constraint_type=ConstraintType.INEQUALITY
        )

        # micro-batch size for T constraint forward
        self.t_batch_size = 1024
        self.margin = 1.0

        # regularization strength on group-mean separation in probability space
        self.group_reg_lambda = 0.1

    def compute_auc_surrogate(self, scores_pos, scores_neg):
        """
        Smooth pairwise surrogate for AUC on *probabilities*.
        Uses sigmoid(m * (p_pos - p_neg)), which is differentiable
        and less sensitive to logit-scale pathologies.
        """
        if scores_pos.numel() == 0 or scores_neg.numel() == 0:
            # no pairs → fall back to 0.5 (uninformative)
            return torch.tensor(0.5, device=self.device, dtype=torch.float32)

        # [P,1] - [1,N] → [P,N]
        pairwise_diff = scores_pos.view(-1, 1) - scores_neg.view(1, -1)
        return torch.sigmoid(self.margin * pairwise_diff).mean()

    def _extract_scores(self, raw_logits):
        """
        Extracts the logit (pre-sigmoid) for the positive class.
        """
        if raw_logits.dim() == 2 and raw_logits.size(-1) == 2:
            return raw_logits[:, 1]  # class-1 logit
        elif raw_logits.dim() == 2 and raw_logits.size(-1) == 1:
            return raw_logits.squeeze(-1)
        else:
            return raw_logits.squeeze()

    def compute_cmp_state(self, model, inputs, targets):
        # ---- forward on D batch (use autocast) ----
        model_inputs = {
            k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            raw = model(**model_inputs).logits

        # extract scalar score per example (logit for positive class)
        score_logits = self._extract_scores(raw).float()  # [B]
        # convert to probabilities; no clamping to avoid artificial compression
        scores = torch.sigmoid(score_logits)              # probs in (0,1)

        groups = inputs["group"].to(self.device).view(-1)
        y = targets.to(self.device).float().view(-1)

        mask0 = (groups == 0)
        mask1 = (groups == 1)

        # per-group positive / negative scores in probability space
        pos0 = scores[mask0 & (y == 1)]
        neg0 = scores[mask0 & (y == 0)]
        pos1 = scores[mask1 & (y == 1)]
        neg1 = scores[mask1 & (y == 0)]

        auc_0 = self.compute_auc_surrogate(pos0, neg0)
        auc_1 = self.compute_auc_surrogate(pos1, neg1)

        # keep sign convention as in your original code:
        # Δ = AUC_g0 - AUC_g1
        auc_gap = auc_0 - auc_1
        loss = -auc_gap if self.maximize else auc_gap  # minimize this
         
        # --- NEW: mild regularization on group mean separation in probability space ---
        reg = 0.0
        if mask0.any() and mask1.any():
          mean_diff = scores[mask0].mean() - scores[mask1].mean()
          reg = self.group_reg_lambda * (mean_diff ** 2)
          loss = loss + reg

      

        # ---- constraint on T: chunked forward to avoid OOM ----
        N_T = self.inputs_T["input_ids"].size(0)
        probs_T_parts = []
        ref_T = torch.clamp(self.constraint_pred.to(self.device).float(), 1e-6, 1 - 1e-6)

        for start in range(0, N_T, self.t_batch_size):
            end = min(start + self.t_batch_size, N_T)
            model_inputs_T = {
                k: v[start:end].to(self.device, non_blocking=True)
                for k, v in self.inputs_T.items()
                if k in ["input_ids", "attention_mask", "token_type_ids"]
            }
            with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                raw_T = model(**model_inputs_T).logits
            logits_T_chunk = self._extract_scores(raw_T).float()
            probs_T_parts.append(torch.sigmoid(logits_T_chunk))
            # free per-chunk tensors ASAP
            del model_inputs_T, raw_T, logits_T_chunk

        probs_T = torch.cat(probs_T_parts, dim=0)  # [|T|]
        # per-example constraint violation in probability space
        per_ex_violation = torch.relu(torch.abs(probs_T - ref_T) - self.lambda_penalty)  # [|T|]
        del probs_T_parts

        constraint_state = ConstraintState(violation=per_ex_violation)
        observed_constraints = {self.constraint: constraint_state}

        return CMPState(loss=loss, observed_constraints=observed_constraints)


# =========================
#   TRAINING (PAIRWISE C-ERM)
# =========================

def train_cerm_pairwise(
    model,
    df_D_mapped,
    df_T_mapped,
    dic_constraint,
    epochs: int,
    batch_size: int,
    lambda_penalty: float,          # constraint tolerance
    tokenizer,
    maximize: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # lighten model footprint
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.output_hidden_states = False
        model.config.output_attentions = False

    # dataset + loader
    df_D_mapped = df_D_mapped.cast_column("id", Value("int64"))
    data_collator = CustomDataCollator(tokenizer)
    dataloader = DataLoader(
        df_D_mapped.with_format("torch"),
        batch_size=batch_size,
        shuffle=True,  # keep ordered if needed
        collate_fn=data_collator,
        num_workers=2,
        persistent_workers=False,
        pin_memory=False,
    )

    # Build T (keep on CPU; chunked to GPU inside CMP)
    ids_T = (
        list(map(int, df_T_mapped["id"]))

        if "id" in df_T_mapped.column_names
        else list(range(len(df_T_mapped["input_ids"])))
    )
    constraint_pred_list = [float(dic_constraint[str(i)]) for i in ids_T]
    constraint_pred = torch.tensor(constraint_pred_list, dtype=torch.float32, device="cpu")

    inputs_T = {
        "input_ids": torch.tensor(df_T_mapped["input_ids"]).long(),
        "attention_mask": torch.tensor(df_T_mapped["attention_mask"]).long(),
    }
    if "token_type_ids" in df_T_mapped.column_names:
        inputs_T["token_type_ids"] = torch.tensor(df_T_mapped["token_type_ids"]).long()

    cmp = PairwiseCERMProblem(model, inputs_T, constraint_pred, lambda_penalty, maximize, device)

    primal_optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dual_optimizer = torch.optim.SGD(cmp.dual_parameters(), lr=1e-2, maximize=True)

    optimizer = SimultaneousOptimizer(
        cmp=cmp, primal_optimizers=primal_optimizer, dual_optimizers=dual_optimizer
    )

    model.train()

    beta = 0.9  # EMA smoothing
    ema_gap = None
    ema_vio = None

    vio_mean_hist, vio_max_hist, lam_mean_hist, lam_max_hist, auc_gap_hist = [], [], [], [], []

    for epoch in range(epochs):
        for batch_i, batch in enumerate(dataloader):
            inputs_D = {
                k: batch[k].to(device)
                for k in ["input_ids", "attention_mask", "group"]
                if k in batch
            }
            if "token_type_ids" in batch:
                inputs_D["token_type_ids"] = batch["token_type_ids"].to(device)
            targets_D = batch["labels"].to(device)

            rollout = optimizer.roll(
                compute_cmp_state_kwargs={"model": model, "inputs": inputs_D, "targets": targets_D}
            )

            # keep λ ≥ 0
            with torch.no_grad():
                cmp.multiplier().data.clamp_(min=0.0)

            cmp_state = rollout.cmp_state
            loss_val = float(cmp_state.loss.detach().cpu())
            auc_gap = abs(loss_val)  # magnitude of surrogate ΔAUC (plus reg, but dominated by gap)

            vio_vec = next(iter(cmp_state.observed_constraints.values())).violation.detach().cpu().float()
            vio_mean = float(vio_vec.mean())
            vio_max = float(vio_vec.max())

            lam_vec = cmp.multiplier().detach().cpu().float()
            lam_mean = float(lam_vec.mean())
            lam_max = float(lam_vec.max())
            try:
                lam_p95 = float(torch.quantile(lam_vec, 0.95))
            except Exception:
                lam_p95 = lam_max

            ema_gap = auc_gap if ema_gap is None else beta * ema_gap + (1 - beta) * auc_gap
            ema_vio = vio_mean if ema_vio is None else beta * ema_vio + (1 - beta) * vio_mean

            auc_gap_hist.append(auc_gap)
            vio_mean_hist.append(vio_mean)
            vio_max_hist.append(vio_max)
            lam_mean_hist.append(lam_mean)
            lam_max_hist.append(lam_max)

            if (batch_i % 10 == 0) or (batch_i == len(dataloader) - 1):
                print(
                    f"[Epoch {epoch+1} | Batch {batch_i+1}/{len(dataloader)}] "
                    f"|ΔAUC_surr|_EMA={ema_gap:.4f}  viō={ema_vio:.4f}  "
                    f"vio_max={vio_max:.4f}  λ̄={lam_mean:.4f}  λ^={lam_max:.4f}  λ_p95={lam_p95:.4f}"
                )

        print(
            f"Epoch {epoch+1} done → |ΔAUC_surr|={auc_gap_hist[-1]:.4f}  "
            f"viō={vio_mean_hist[-1]:.4f}  vio_max={vio_max_hist[-1]:.4f}  "
            f"λ̄={lam_mean_hist[-1]:.4f}  λ^={lam_max_hist[-1]:.4f}"
        )

        # optional early stop when (almost) feasible for a while
        if len(vio_mean_hist) > 10 and np.mean(vio_mean_hist[-10:]) < 1e-3:
            print("Early stop: constraint mean violation < 1e-3 for last 10 steps.")
            break

    print("Training done for C-ERM")
    return model


# =========================
#   SURROGATE VS TRUE AUC HELPERS
# =========================

@torch.no_grad()
def predict_logits_batched(
    h,
    inputs_D: Dict[str, torch.Tensor],
    batch_size: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Batched inference that returns raw class-1 logits [N],
    matching the representation used inside PairwiseCERMProblem.
    """
    keys = [k for k in ["input_ids", "attention_mask"] if k in inputs_D]
    if "token_type_ids" in inputs_D:
        keys.append("token_type_ids")

    # ensure CPU base tensors
    N = None
    tensors_cpu: Dict[str, torch.Tensor] = {}
    for k in keys:
        arr = inputs_D[k]
        if not isinstance(arr, torch.Tensor):
            arr = torch.as_tensor(arr)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        tensors_cpu[k] = arr.cpu()
        N = arr.size(0) if N is None else N
    assert all(tensors_cpu[k].size(0) == N for k in keys), "Input dict lengths mismatch."

    h = h.to(device)
    h.eval()

    logits_all: List[torch.Tensor] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = {k: v[start:end].to(device, non_blocking=True) for k, v in tensors_cpu.items()}
        outputs = h(**batch)
        raw_logits = outputs.logits
        if raw_logits.dim() == 2 and raw_logits.size(-1) == 2:
            batch_logits = raw_logits[:, 1]
        elif raw_logits.dim() == 2 and raw_logits.size(-1) == 1:
            batch_logits = raw_logits.squeeze(-1)
        else:
            batch_logits = raw_logits.squeeze()
        logits_all.append(batch_logits.detach().cpu())
        del batch, outputs, raw_logits, batch_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(logits_all, dim=0)  # [N]


def _pairwise_auc_surrogate_from_probs(
    probs_pos: torch.Tensor,
    probs_neg: torch.Tensor,
    margin: float = 1.0,
) -> float:
    """
    Pairwise surrogate AUC computed in probability space.
    This matches PairwiseCERMProblem.compute_auc_surrogate.
    """
    if probs_pos.numel() == 0 or probs_neg.numel() == 0:
        return float("nan")
    pairwise_diff = probs_pos.view(-1, 1) - probs_neg.view(1, -1)
    val = torch.sigmoid(margin * pairwise_diff).mean()
    return float(val.detach().cpu())



@torch.no_grad()
def evaluate_pairwise_surrogate_vs_true_auc_gap_on_D(
    logits: torch.Tensor,
    df_D: pd.DataFrame,
    group0: int = 0,
    group1: int = 1,
    margin: float = 1.0,
    device: str = "cpu",
) -> Dict[str, float]:
    logits = logits.to(device).view(-1)

    if "true_label" not in df_D.columns:
        raise ValueError("df_D must contain 'true_label'.")
    if "group" not in df_D.columns:
        raise ValueError("df_D must contain 'group'.")

    labels_np = df_D["true_label"].astype(int).to_numpy()
    groups_np = df_D["group"].astype(int).to_numpy()

    labels = torch.as_tensor(labels_np, device=device, dtype=torch.long)
    groups = torch.as_tensor(groups_np, device=device, dtype=torch.long)

    probs = torch.sigmoid(logits)

    def _per_group(g: int):
        mask = (groups == g)
        y = labels[mask]
        p = probs[mask]

        if y.unique().numel() < 2:
            auc_true = float("nan")
        else:
            auc_true = roc_auc_score(
                y.detach().cpu().numpy().astype(int),
                p.detach().cpu().numpy().astype(float),
            )

        pos = p[y == 1]
        neg = p[y == 0]
        auc_surr = _pairwise_auc_surrogate_from_probs(pos, neg, margin=margin)
        return auc_true, auc_surr

    auc_true_g0, auc_surr_g0 = _per_group(group0)
    auc_true_g1, auc_surr_g1 = _per_group(group1)

    delta_true = auc_true_g0 - auc_true_g1
    delta_surr = auc_surr_g0 - auc_surr_g1

    return {
        "auc_true_g0": float(auc_true_g0),
        "auc_true_g1": float(auc_true_g1),
        "delta_true": float(delta_true),
        "auc_surr_g0": float(auc_surr_g0),
        "auc_surr_g1": float(auc_surr_g1),
        "delta_surr": float(delta_surr),
        "delta_gap": float(delta_surr - delta_true),
    }


# =========================
#   TOP-LEVEL EVAL (MAX/MIN) — returns SURROGATE ΔAUC
# =========================

def eval_h(
    base_model_factory: Callable,   # pass a factory, not a model instance
    df_D: pd.DataFrame,
    df_D_mapped,
    inputs_D: Dict[str, torch.Tensor],
    df_T_mapped,
    constraint_pred: Dict[str, float],
    epochs_opt: int,
    batch_size: int,
    lambda_penalty: float,
    tokenizer,
    Maximize: bool,
    compute_group_auc_diff_fn: Callable,
) -> Tuple[np.ndarray, float]:
    """
    Trains a fresh surrogate via C-ERM and evaluates ΔAUC on full D.

    IMPORTANT CHANGE:
    ------------------
    - We now return the *surrogate* ΔAUC (delta_surr) as the second output.
      This is what you will use as the bound for next-sample selection.

    Returns:
        scores_array (probs on D), delta_surr (pairwise-surrogate-based ΔAUC)
    """
    print(f'Training CERM pairwise ({"MAX" if Maximize else "MIN"})')

    # Train fresh model (no leakage across max/min)
    h = train_cerm_pairwise(
        base_model_factory(),
        df_D_mapped,
        df_T_mapped,
        constraint_pred,
        epochs_opt,
        batch_size,
        lambda_penalty,
        tokenizer,
        maximize=Maximize,
    )

    print("Computing logits and AUC on full dataset D...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        # 1) logits on D (for surrogate metric)
        logits = predict_logits_batched(
            h,
            inputs_D,
            batch_size=batch_size,
            device=device,
        )
        # 2) probabilities on D (for potential debugging / ROC metrics)
        scores = torch.sigmoid(logits).cpu().numpy()
    print("Done computing logits/probs on D")

    # Debug: compute true ROC AUC-based ΔAUC (not used for bounds)
    with torch.no_grad():
        delta_true, _ = compute_group_auc_difference_from_scores(
            scores,
            df_D,
            group1=1,
            group2=0,
            device="cpu"
        )

    # Evaluate pairwise surrogate vs true ROC AUC
    surrogate_eval = evaluate_pairwise_surrogate_vs_true_auc_gap_on_D(
        logits=logits,
        df_D=df_D,
        group0=0,
        group1=1,
        margin=1.0,
        device="cpu",
    )
    print(f"[{'MAX' if Maximize else 'MIN'}] pairwise surrogate vs true ROC AUC: {surrogate_eval}")
    print(f"    (debug) Δ_true = {delta_true:.4f}")

    return scores, h


# =========================
#   OPTIONAL: GRADIENT PROBES
# =========================

def compute_lora_gradient(model, texts, labels, groups, tokenizer):
    """
    Per-example gradient probes on LoRA params with a pairwise AUC-style loss.
    This still uses logits, but only for gradient diagnostics, not for the main
    ΔAUC estimation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = torch.tensor(labels).float().to(device)
    groups = torch.tensor(groups).long().to(device)

    with amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        logits = model(**encoded).logits.squeeze()
    logits = logits.float()

    influence_vectors = []
    N = len(texts)
    indices = torch.arange(N, device=device)

    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora" in n]

    for i in range(N):
        model.zero_grad(set_to_none=True)

        group_i = groups[i]
        label_i = labels[i]
        logit_i = logits[i]

        group_mask = (groups == group_i)
        exclude_self = indices != i
        valid_mask = group_mask & exclude_self

        if label_i == 1:
            pos = logit_i.view(-1, 1)
            neg = logits[(labels == 0) & valid_mask].view(1, -1)
        elif label_i == 0:
            pos = logits[(labels == 1) & valid_mask].view(-1, 1)
            neg = logit_i.view(1, -1)
        else:
            influence_vectors.append(torch.zeros(1))
            continue

        if pos.numel() == 0 or neg.numel() == 0:
            influence_vectors.append(torch.zeros(1))
            continue

        pairwise_diff = pos - neg
        loss_i = 1.0 - torch.sigmoid(pairwise_diff).mean()

        grads = torch.autograd.grad(
            outputs=loss_i,
            inputs=[p for (_, p) in lora_params],
            retain_graph=False,
            allow_unused=True
        )

        grad_flat = []
        for (n, p), g in zip(lora_params, grads):
            if g is not None:
                grad_flat.append(g.contiguous().view(-1))
            else:
                grad_flat.append(torch.zeros_like(p).view(-1))
        grad_vector = torch.cat(grad_flat) if len(grad_flat) else torch.zeros(1, device=device)

        influence_vectors.append(grad_vector.detach().cpu())

    # Cleanup
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    gc.collect()

    return influence_vectors
