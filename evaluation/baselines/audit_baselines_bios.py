#!/usr/bin/env python3
"""
Audit Baselines for Bias-in-Bios Dataset - Reproducible Script

This script evaluates ordered sampling strategies (random, stratified, power, BO)
for auditing black-box model fairness on the Bias-in-Bios dataset using a CSV
of precomputed profession classification scores.

Key differences from Jigsaw version:
- Uses BiasInBiosBlackBox (CSV-based) instead of training fresh models
- "Text" is actually ID strings (e.g., "ID123")
- Binary classification: one-vs-rest for target profession
- Embeddings based on actual biography text from HuggingFace dataset

IMPORTANT FIXES (Jan 2026):
- Preserve D indices everywhere (no ignore_index=True in sampling/concat)
- Track already-sampled rows correctly
- Ensure embeddings align to D without dropping rows
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def setup_project_paths():
    """Set up project paths and working directory."""
    project_root = Path.cwd().parent.parent
    if not project_root.exists():
        project_root = Path.cwd()

    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    return project_root


# Set up paths first
project_root = setup_project_paths()

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Import BiasInBiosBlackBox after path setup
from blackboxes.blackbox_api_bias_in_bios import BiasInBiosBlackBox, load_bias_in_bios_with_ids


def set_gpu(gpu_id: Optional[int]) -> None:
    """Set GPU before importing torch-heavy modules."""
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# AUEC Computation (same as Jigsaw version)
# =============================================================================

def area_under_error_curve(
    df: pd.DataFrame,
    t_col: str = "t_size",
    err_col: str = "abs_error_roc_auc_diff",
    t_max: int = 1000,
    prepend_t0: bool = True,
    t0_value: float = 0,
) -> float:
    """Compute AUEC = ∫_0^{t_max} error(t) dt using trapezoidal rule."""
    if df.empty:
        raise ValueError("df is empty")

    d = df[[t_col, err_col]].dropna().copy()
    d = d.sort_values(t_col).drop_duplicates(subset=[t_col], keep="last")
    d = d[d[t_col] <= t_max]

    if d.empty:
        raise ValueError(f"No rows with {t_col} <= t_max ({t_max})")

    if prepend_t0 and (d[t_col].iloc[0] > t0_value):
        first_err = float(d[err_col].iloc[0])
        d = pd.concat(
            [pd.DataFrame({t_col: [t0_value], err_col: [first_err]}), d],
            ignore_index=True,
        )

    if d[t_col].iloc[-1] < t_max:
        last_err = float(d[err_col].iloc[-1])
        d = pd.concat(
            [d, pd.DataFrame({t_col: [t_max], err_col: [last_err]})],
            ignore_index=True,
        )

    t = d[t_col].to_numpy(dtype=float)
    e = d[err_col].to_numpy(dtype=float)

    return float(np.trapz(e, t))


def normalized_area_under_error_curve(
    df: pd.DataFrame,
    t_col: str = "t_size",
    err_col: str = "abs_error_roc_auc_diff",
    t_max: int = 1000,
    **kwargs,
) -> float:
    """Normalized AUEC = (1 / t_max) ∫_0^{t_max} error(t) dt"""
    auec = area_under_error_curve(df, t_col=t_col, err_col=err_col, t_max=t_max, **kwargs)
    return float(auec / float(t_max))


# =============================================================================
# ΔAUC Computation
# =============================================================================

def compute_delta_auc(
    df: pd.DataFrame,
    score_col: str = "bb_score",
    label_col: str = "true_label",
    group_col: str = "group",
    ref_group: int = 0,
    comp_group: int = 1,
) -> float:
    """
    Compute ΔAUC = AUC(ref_group) - AUC(comp_group).
    Returns NaN if either group has only one label class.
    """
    df_ref = df[df[group_col] == ref_group]
    df_comp = df[df[group_col] == comp_group]

    if len(df_ref[label_col].unique()) < 2:
        return np.nan
    if len(df_comp[label_col].unique()) < 2:
        return np.nan

    try:
        auc_ref = roc_auc_score(df_ref[label_col], df_ref[score_col])
        auc_comp = roc_auc_score(df_comp[label_col], df_comp[score_col])
        return float(auc_ref - auc_comp)
    except ValueError:
        return np.nan


# =============================================================================
# BiasInBios Score API Wrapper
# =============================================================================

class BiasInBiosScoreAPI:
    """
    Makes BiasInBiosBlackBox look like the Jigsaw API:
      api.predict_scores(list_of_ids) -> np.array(scores)

    Here, "texts" are actually IDs (strings like "ID123").
    Returns the score for the positive label (target profession).
    """

    def __init__(self, bb: BiasInBiosBlackBox, pos_label: str):
        self.bb = bb
        self.pos_label = str(pos_label)

    def predict_scores(self, ids: List[str], batch_size: int = 4096) -> np.ndarray:
        """Return scores for positive label for given IDs."""
        if isinstance(ids, str):
            return np.array([self.bb.get_score_for_label(ids, self.pos_label)], dtype=np.float32)

        ids = list(ids)
        out = np.empty(len(ids), dtype=np.float32)

        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            for j, sid in enumerate(chunk):
                out[i + j] = self.bb.get_score_for_label(str(sid), self.pos_label)

        return out


# =============================================================================
# Data Loading
# =============================================================================

def build_bios_audit_dataset(
    scores_csv: str,
    pos_label: str,
    n_sample: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, BiasInBiosScoreAPI, BiasInBiosBlackBox]:
    """
    Build audit dataset D from BiasInBios scores CSV.

    Returns DataFrame with columns:
      - id: ID string (e.g., "ID123")
      - text: same as id (for compatibility with sampling functions)
      - true_label: binary (1 if gold_occupation == pos_label, else 0)
      - group: gender (0 or 1)
      - gold_occupation: original profession string
    """
    bb = BiasInBiosBlackBox(scores_csv, verbose=True)

    rows = []
    for sid in bb.ids:
        gold = bb.get_gold_label(sid)
        gender = bb.get_gender(sid)

        if gold is None or gender is None:
            continue

        y = 1 if str(gold) == str(pos_label) else 0
        rows.append({
            "id": str(sid),
            "text": str(sid),  # IMPORTANT: reuse "text" for compatibility
            "true_label": int(y),
            "group": int(gender),
            "gold_occupation": str(gold),
        })

    # Use a stable RangeIndex once; DO NOT reset indices later in sampling code.
    D = pd.DataFrame(rows).reset_index(drop=True)

    # Sample if requested (this resets D indices ON PURPOSE once)
    if n_sample is not None and n_sample < len(D):
        D = D.sample(n=n_sample, random_state=seed).reset_index(drop=True)
        print(f"Sampled {n_sample} rows from {len(rows)} total")

    print(f"Audit dataset: {len(D)} rows")
    print(f"Target profession: {pos_label}")
    print(f"Label distribution: {D['true_label'].value_counts().to_dict()}")
    print(f"Group distribution: {D['group'].value_counts().to_dict()}")

    api = BiasInBiosScoreAPI(bb, pos_label=pos_label)
    return D, api, bb


def load_bios_texts_for_embeddings(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load biography texts from HuggingFace for computing embeddings.

    IMPORTANT FIX:
    - Do NOT drop rows (dropping breaks alignment with D / bb_scores_full).
    - Missing texts become "".
    """
    print("Loading biography texts from HuggingFace...")
    hf_df = load_bias_in_bios_with_ids()

    id_to_text = dict(zip(hf_df["id"], hf_df["hard_text"]))

    out = audit_df.copy()
    out["bio_text"] = out["id"].map(id_to_text)

    missing = out["bio_text"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} IDs have no corresponding biography text. Filling with empty string.")
        out["bio_text"] = out["bio_text"].fillna("")

    return out


# =============================================================================
# Sampling Utilities
# =============================================================================

def stratified_sampling(n: int, df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    """
    Stratified sampling: sample n rows proportional to group sizes.

    IMPORTANT FIX:
    - Preserve original indices (NO ignore_index=True).
    """
    group_counts = df[group_col].value_counts()
    n_per_group = {g: int(np.ceil(n * count / len(df))) for g, count in group_counts.items()}

    total = sum(n_per_group.values())
    if total > n:
        sorted_groups = sorted(n_per_group.items(), key=lambda x: -x[1])
        for g, _ in sorted_groups:
            if total <= n:
                break
            if n_per_group[g] > 1:
                n_per_group[g] -= 1
                total -= 1

    samples = []
    for group_val, n_sample in n_per_group.items():
        group_df = df[df[group_col] == group_val]
        if len(group_df) >= n_sample:
            samples.append(group_df.sample(n=n_sample, replace=False))
        else:
            samples.append(group_df)

    # Preserve D indices
    return pd.concat(samples)


def _available_index(D: pd.DataFrame, already_sampled_indices: set) -> pd.Index:
    """Helper: compute available indices robustly."""
    if not already_sampled_indices:
        return D.index
    return D.index.difference(pd.Index(list(already_sampled_indices)))


def random_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
) -> pd.DataFrame:
    """Random sampling of k new rows from D not in S."""
    available = _available_index(D, already_sampled_indices)
    if len(available) == 0:
        return pd.DataFrame()

    k_actual = min(k, len(available))
    sampled_indices = np.random.choice(available.to_numpy(), size=k_actual, replace=False)
    return D.loc[sampled_indices]


def stratified_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
    group_col: str = "group",
) -> pd.DataFrame:
    """Stratified sampling of k new rows from D not in S."""
    available = _available_index(D, already_sampled_indices)
    if len(available) == 0:
        return pd.DataFrame()

    available_df = D.loc[available]
    return stratified_sampling(min(k, len(available_df)), available_df, group_col=group_col)


def power_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
    bb_scores: np.ndarray,
    gamma: float = 2.0,
) -> pd.DataFrame:
    """
    Power sampling: sample based on uncertainty proxy p(1-p) raised to power gamma.

    IMPORTANT FIXES:
    - Use correct D indices (works now that indices are preserved).
    - If weights sum to 0 (all p near 0/1), fall back to uniform sampling.
    """
    available = _available_index(D, already_sampled_indices)
    if len(available) == 0:
        return pd.DataFrame()

    # Map available indices to positions in bb_scores_full (bb_scores_full is aligned with D order)
    # Since D is RangeIndex 0..N-1, idx == position; but keep it general:
    available_positions = [D.index.get_loc(idx) for idx in available]
    available_scores = bb_scores[np.array(available_positions, dtype=int)]

    p = np.clip(available_scores, 1e-6, 1 - 1e-6)
    uncertainty = p * (1 - p)

    weights = np.power(uncertainty, gamma)
    wsum = float(weights.sum())

    k_actual = min(k, len(available))
    if not np.isfinite(wsum) or wsum <= 0.0:
        # Uniform fallback
        sampled_indices = np.random.choice(available.to_numpy(), size=k_actual, replace=False)
        return D.loc[sampled_indices]

    weights = weights / wsum
    sampled_positions = np.random.choice(len(available), size=k_actual, replace=False, p=weights)
    sampled_indices = available.to_numpy()[sampled_positions]

    return D.loc[sampled_indices]


def bo_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
    bb_scores: np.ndarray,
    text_embeddings: np.ndarray,
    true_delta: float,
    xi: float = 0.01,
) -> pd.DataFrame:
    """
    Bayesian optimization sampling using GP surrogate with EI acquisition.

    NOTE:
    This is still a *baseline* BO and uses a very weak target.
    The important fix here is index alignment: S.index must be D indices.
    """
    available = _available_index(D, already_sampled_indices)
    if len(available) == 0:
        return pd.DataFrame()

    # Training inputs = embeddings of sampled points
    S_positions = [D.index.get_loc(idx) for idx in S.index]
    X_train = text_embeddings[np.array(S_positions, dtype=int)]

    current_delta = compute_delta_auc(S)
    if np.isnan(current_delta):
        current_delta = 0.0

    # Weak BO target (kept as-is for baseline)
    y_train = np.array([-abs(current_delta - true_delta)] * len(S), dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)

    try:
        gp.fit(X_train_scaled, y_train)
    except Exception:
        k_actual = min(k, len(available))
        sampled_indices = np.random.choice(available.to_numpy(), size=k_actual, replace=False)
        return D.loc[sampled_indices]

    available_positions = [D.index.get_loc(idx) for idx in available]
    X_pool = text_embeddings[np.array(available_positions, dtype=int)]
    X_pool_scaled = scaler.transform(X_pool)

    mu, sigma = gp.predict(X_pool_scaled, return_std=True)
    sigma = np.maximum(sigma, 1e-6)

    best_y = y_train.max()
    z = (mu - best_y - xi) / sigma
    ei = (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-6] = 0.0

    k_actual = min(k, len(available))
    top_k_positions = np.argsort(-ei)[:k_actual]
    sampled_indices = available.to_numpy()[top_k_positions]

    return D.loc[sampled_indices]


# =============================================================================
# Text Embeddings (for BO strategy)
# =============================================================================

def compute_text_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: Optional[str] = None,
    max_length: int = 256,
) -> np.ndarray:
    """Compute sentence embeddings for biography texts."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing embeddings on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch = texts[i:i + batch_size]
            tokens = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)

            outputs = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp(min=1)

            embeddings.append(pooled.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# =============================================================================
# Main Audit Loop (Bias-in-Bios)
# =============================================================================

def run_single_seed_audit(
    seed: int,
    D: pd.DataFrame,
    api: BiasInBiosScoreAPI,
    max_queries: int,
    k_init: int,
    k_batch: int,
    strategy: str,
    true_delta: float,
    text_embeddings: Optional[np.ndarray] = None,
    bb_scores_full: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run audit for a single seed and strategy."""
    set_all_seeds(seed)

    # Initialize with stratified sampling (preserves D indices)
    S = stratified_sampling(k_init, D).copy()
    S["bb_score"] = api.predict_scores(S["text"].tolist())

    already_sampled = set(S.index)  # D indices (correct now)
    trajectory = []

    # Record initial point
    current_delta = compute_delta_auc(S)
    if not np.isnan(current_delta):
        trajectory.append({
            "t_size": len(S),
            "delta_auc": current_delta,
            "abs_error_roc_auc_diff": abs(current_delta - true_delta),
        })

    n_batches = max(0, (max_queries - k_init) // k_batch)

    for _ in tqdm(range(n_batches), desc=f"Seed {seed}, {strategy}", leave=False):
        if len(already_sampled) >= len(D):
            break

        if strategy == "random":
            new_batch = random_ordered_sampling(k_batch, S, D, already_sampled)

        elif strategy == "stratified":
            new_batch = stratified_ordered_sampling(k_batch, S, D, already_sampled)

        elif strategy == "power":
            if bb_scores_full is None:
                raise ValueError("bb_scores_full required for power sampling")
            new_batch = power_ordered_sampling(
                k_batch, S, D, already_sampled, bb_scores_full, gamma=2.0
            )

        elif strategy == "bo":
            if text_embeddings is None or bb_scores_full is None:
                raise ValueError("text_embeddings and bb_scores_full required for BO")
            new_batch = bo_ordered_sampling(
                k_batch, S, D, already_sampled, bb_scores_full, text_embeddings, true_delta
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(new_batch) == 0:
            break

        # Query BB for new batch
        new_batch = new_batch.copy()
        new_batch["bb_score"] = api.predict_scores(new_batch["text"].tolist())

        # Update sampled set BEFORE concat (paranoia)
        already_sampled.update(new_batch.index)

        # IMPORTANT FIX: preserve indices (NO ignore_index=True)
        S = pd.concat([S, new_batch])

        current_delta = compute_delta_auc(S)
        if not np.isnan(current_delta):
            trajectory.append({
                "t_size": len(S),
                "delta_auc": current_delta,
                "abs_error_roc_auc_diff": abs(current_delta - true_delta),
            })

    return pd.DataFrame(trajectory)


def run_multi_seed_audit(
    seeds: List[int],
    scores_csv: str,
    pos_label: str,
    n_audit: Optional[int],
    max_queries: int,
    k_init: int,
    k_batch: int,
    strategies: List[str],
    out_dir: Path,
    device: str = "cuda",
) -> None:
    """Run audit across multiple seeds and strategies."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading Bias-in-Bios black-box from {scores_csv}")
    print(f"Target profession: {pos_label}")
    print(f"{'='*60}")

    D, api, _bb = build_bios_audit_dataset(
        scores_csv=scores_csv,
        pos_label=pos_label,
        n_sample=n_audit,
        seed=42,  # Fixed seed for dataset sampling
    )

    # True ΔAUC on full audit set
    print("\nComputing true ΔAUC on full audit set...")
    D_full = D.copy()
    D_full["bb_score"] = api.predict_scores(D_full["text"].tolist())
    true_delta = compute_delta_auc(D_full)
    print(f"True ΔAUC = {true_delta:.4f}")

    bb_scores_full = None
    text_embeddings = None

    if "power" in strategies or "bo" in strategies:
        print("\nPrecomputing black-box scores for full dataset...")
        bb_scores_full = api.predict_scores(D["text"].tolist())

    if "bo" in strategies:
        print("\nLoading biography texts and computing embeddings...")
        D_with_texts = load_bios_texts_for_embeddings(D)
        text_embeddings = compute_text_embeddings(
            D_with_texts["bio_text"].tolist(),
            device=device,
        )
        assert len(text_embeddings) == len(D), "Embeddings length mismatch"

    all_trajectories = {strategy: [] for strategy in strategies}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")

        set_all_seeds(seed)

        for strategy in strategies:
            print(f"\nRunning strategy: {strategy}")

            trajectory_df = run_single_seed_audit(
                seed=seed,
                D=D,
                api=api,
                max_queries=max_queries,
                k_init=k_init,
                k_batch=k_batch,
                strategy=strategy,
                true_delta=true_delta,
                text_embeddings=text_embeddings,
                bb_scores_full=bb_scores_full,
            )

            trajectory_df["seed"] = seed
            trajectory_df["strategy"] = strategy
            trajectory_df["true_delta"] = true_delta
            trajectory_df["pos_label"] = pos_label

            seed_file = out_dir / f"trajectory_{strategy}_seed{seed}.csv"
            trajectory_df.to_csv(seed_file, index=False)
            print(f"Saved: {seed_file}")

            all_trajectories[strategy].append(trajectory_df)

    print(f"\n{'='*60}")
    print("Saving combined trajectories...")
    for strategy, traj_list in all_trajectories.items():
        combined_df = pd.concat(traj_list, ignore_index=True)
        combined_file = out_dir / f"trajectory_{strategy}_all.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"Saved: {combined_file}")


# =============================================================================
# Summary Statistics (same as your version)
# =============================================================================

def compute_summary_statistics(
    out_dir: Path,
    strategies: List[str],
    eps_thresholds: List[float],
    max_queries: int,
    target_query_budget: int = 250,
) -> pd.DataFrame:
    """Compute summary statistics across all seeds for each strategy."""
    summary_rows = []

    for strategy in strategies:
        combined_file = out_dir / f"trajectory_{strategy}_all.csv"
        if not combined_file.exists():
            print(f"Warning: {combined_file} not found, skipping {strategy}")
            continue

        df_all = pd.read_csv(combined_file)
        seeds = df_all["seed"].unique()

        summary_df = (
            df_all.groupby("t_size")["abs_error_roc_auc_diff"]
            .agg(["mean", "std", "count", "median"])
            .reset_index()
        )

        for eps in eps_thresholds:
            crossed = summary_df[summary_df["mean"] <= eps]
            mean_crossing_T = float(crossed["t_size"].min()) if len(crossed) > 0 else np.nan

            if not np.isnan(mean_crossing_T):
                at_crossing = summary_df[summary_df["t_size"] == mean_crossing_T].iloc[0]
                error_at_crossing_mean = float(at_crossing["mean"])
                error_at_crossing_std = float(at_crossing["std"])
                error_at_crossing_median = float(at_crossing["median"])
                n_at_crossing = int(at_crossing["count"])
            else:
                error_at_crossing_mean = np.nan
                error_at_crossing_std = np.nan
                error_at_crossing_median = np.nan
                n_at_crossing = 0

            seed_crossing_times = []
            for seed in seeds:
                df_seed = df_all[df_all["seed"] == seed].copy().sort_values("t_size")
                below_eps = df_seed[df_seed["abs_error_roc_auc_diff"] <= eps]
                if len(below_eps) > 0:
                    seed_crossing_times.append(float(below_eps["t_size"].iloc[0]))
                else:
                    seed_crossing_times.append(np.nan)

            valid_crossings = [t for t in seed_crossing_times if not np.isnan(t)]
            queries_to_eps_mean = np.mean(valid_crossings) if valid_crossings else np.nan
            queries_to_eps_std = np.std(valid_crossings) if valid_crossings else np.nan
            queries_to_eps_median = np.median(valid_crossings) if valid_crossings else np.nan
            n_converged = len(valid_crossings)

            row = {
                "strategy": strategy,
                "epsilon": eps,
                "mean_crossing_T": mean_crossing_T,
                "error_at_crossing_T_mean": error_at_crossing_mean,
                "error_at_crossing_T_std": error_at_crossing_std,
                "error_at_crossing_T_median": error_at_crossing_median,
                "n_at_crossing_T": n_at_crossing,
                f"queries_to_{eps}_mean": queries_to_eps_mean,
                f"queries_to_{eps}_std": queries_to_eps_std,
                f"queries_to_{eps}_median": queries_to_eps_median,
                f"queries_to_{eps}_n_reached": n_converged,
            }
            summary_rows.append(row)

        auec_values = []
        error_at_target = []

        for seed in seeds:
            df_seed = df_all[df_all["seed"] == seed].copy().sort_values("t_size")

            try:
                auec = normalized_area_under_error_curve(
                    df_seed,
                    t_col="t_size",
                    err_col="abs_error_roc_auc_diff",
                    t_max=max_queries,
                    prepend_t0=True,
                )
                auec_values.append(auec)
            except Exception as e:
                print(f"Warning: AUEC failed for {strategy} seed {seed}: {e}")
                auec_values.append(np.nan)

            if target_query_budget in df_seed["t_size"].values:
                error_at_target.append(
                    df_seed[df_seed["t_size"] == target_query_budget]["abs_error_roc_auc_diff"].iloc[0]
                )
            else:
                before = df_seed[df_seed["t_size"] <= target_query_budget]
                after = df_seed[df_seed["t_size"] >= target_query_budget]
                if len(before) > 0 and len(after) > 0:
                    t1, e1 = before["t_size"].iloc[-1], before["abs_error_roc_auc_diff"].iloc[-1]
                    t2, e2 = after["t_size"].iloc[0], after["abs_error_roc_auc_diff"].iloc[0]
                    if t1 == t2:
                        error_interp = e1
                    else:
                        error_interp = e1 + (e2 - e1) * (target_query_budget - t1) / (t2 - t1)
                    error_at_target.append(error_interp)
                else:
                    error_at_target.append(np.nan)

        auec_clean = [v for v in auec_values if not np.isnan(v)]
        auec_mean = np.mean(auec_clean) if auec_clean else np.nan
        auec_std = np.std(auec_clean) if auec_clean else np.nan

        error_clean = [v for v in error_at_target if not np.isnan(v)]
        error_at_target_mean = np.mean(error_clean) if error_clean else np.nan
        error_at_target_std = np.std(error_clean) if error_clean else np.nan

        for row in summary_rows:
            if row["strategy"] == strategy:
                row["auec_mean"] = auec_mean
                row["auec_std"] = auec_std
                row[f"error_at_{target_query_budget}_mean"] = error_at_target_mean
                row[f"error_at_{target_query_budget}_std"] = error_at_target_std

    summary_df = pd.DataFrame(summary_rows)

    col_order = ["strategy", "epsilon"]
    col_order.extend([c for c in summary_df.columns if "crossing_T" in c and c not in col_order])
    col_order.extend([c for c in summary_df.columns if "queries_to" in c and c not in col_order])
    col_order.extend([c for c in summary_df.columns if c not in col_order])
    summary_df = summary_df[col_order]

    summary_file = out_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary: {summary_file}")

    return summary_df


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit baselines for Bias-in-Bios dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID (None for CPU)")
    parser.add_argument("--max_queries", type=int, default=2000, help="Maximum query budget")
    parser.add_argument("--k_init", type=int, default=4, help="Initial stratified sample size")
    parser.add_argument("--k_batch", type=int, default=16, help="Batch size for ordered sampling")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated seeds")
    parser.add_argument("--eps", type=str, default="0.02,0.05", help="Comma-separated error thresholds")
    parser.add_argument("--strategies", type=str, default="random,stratified,power,bo", help="Comma-separated strategies")
    parser.add_argument("--scores_csv", type=str, default="blackboxes/blackbox_bios.csv", help="Scores CSV path")
    parser.add_argument("--pos_label", type=str, default="professor", help="One-vs-rest target profession")
    parser.add_argument("--n_audit", type=int, default=None, help="Audit dataset size (None=all)")
    parser.add_argument("--out_dir", type=str, default="evaluation/baselines/results_bios", help="Output directory")
    parser.add_argument("--target_query_budget", type=int, default=250, help="Query budget for error metric")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU before importing torch-heavy modules
    set_gpu(args.gpu)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    eps_thresholds = [float(e.strip()) for e in args.eps.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    out_dir = Path(args.out_dir)

    print("=" * 60)
    print("Audit Baselines - Bias-in-Bios Dataset")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"Strategies: {strategies}")
    print(f"Max queries: {args.max_queries}")
    print(f"Initial sample size: {args.k_init}")
    print(f"Batch size: {args.k_batch}")
    print(f"Scores CSV: {args.scores_csv}")
    print(f"Target profession: {args.pos_label}")
    print(f"Audit dataset size: {args.n_audit if args.n_audit else 'all'}")
    print(f"Error thresholds: {eps_thresholds}")
    print(f"Output directory: {out_dir}")
    print(f"GPU: {args.gpu if args.gpu is not None else 'CPU'}")
    print("=" * 60)

    import torch
    device = "cuda" if torch.cuda.is_available() and args.gpu is not None else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    start_time = time.time()

    run_multi_seed_audit(
        seeds=seeds,
        scores_csv=args.scores_csv,
        pos_label=args.pos_label,
        n_audit=args.n_audit,
        max_queries=args.max_queries,
        k_init=args.k_init,
        k_batch=args.k_batch,
        strategies=strategies,
        out_dir=out_dir,
        device=device,
    )

    print(f"\n{'='*60}")
    print("Computing summary statistics...")
    summary_df = compute_summary_statistics(
        out_dir=out_dir,
        strategies=strategies,
        eps_thresholds=eps_thresholds,
        max_queries=args.max_queries,
        target_query_budget=args.target_query_budget,
    )

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
