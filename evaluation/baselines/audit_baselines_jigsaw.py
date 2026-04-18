#!/usr/bin/env python3
"""
Audit Baselines for Jigsaw Dataset - Reproducible Script

This script evaluates ordered sampling strategies (random, stratified, power, BO)
for auditing black-box model fairness on the Jigsaw toxic comment dataset.

Each seed trains a fresh black-box model on SBIC, then audits it on Jigsaw
using various sampling strategies to estimate ΔAUC (group AUC difference).
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


def setup_project_paths():
    """Set up project paths and working directory."""
    # Add project root to Python path (assuming we're in a subfolder)
    project_root = Path.cwd().parent.parent
    if not project_root.exists():
        # Fallback: try current directory
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
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Import data loaders after path setup
from data_loader import load_jigsaw, load_sbic_and_train_api_df


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
# AUEC Computation
# =============================================================================

def area_under_error_curve(
    df: pd.DataFrame,
    t_col: str = "t_size",
    err_col: str = "abs_error_roc_auc_diff",
    t_max: int = 1000,
    prepend_t0: bool = True,
    t0_value: float = 0,
) -> float:
    """
    Compute AUEC = ∫_0^{t_max} error(t) dt using trapezoidal rule.
    
    Args:
        df: DataFrame with trajectory data
        t_col: Column name for query budget/size
        err_col: Column name for absolute error
        t_max: Maximum query budget
        prepend_t0: Whether to prepend t=0 with first error value
        t0_value: Value for t=0
        
    Returns:
        Area under error curve
    """
    if df.empty:
        raise ValueError("df is empty")
    
    d = df[[t_col, err_col]].dropna().copy()
    d = d.sort_values(t_col).drop_duplicates(subset=[t_col], keep="last")
    d = d[d[t_col] <= t_max]
    
    if d.empty:
        raise ValueError(f"No rows with {t_col} <= t_max ({t_max})")
    
    # Prepend t=0 if needed
    if prepend_t0 and (d[t_col].iloc[0] > t0_value):
        first_err = float(d[err_col].iloc[0])
        d = pd.concat(
            [pd.DataFrame({t_col: [t0_value], err_col: [first_err]}), d],
            ignore_index=True,
        )
    
    # Ensure point at t_max (carry forward last error)
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
    label_col: str = "label",
    group_col: str = "group",
    ref_group: int = 0,  # white
    comp_group: int = 1,  # black
) -> float:
    """
    Compute ΔAUC = AUC(ref_group) - AUC(comp_group) = AUC(white) - AUC(black).
    
    Convention: white=0 (reference), black=1 (comparison)
    Positive ΔAUC means model performs better on white group.
    
    Returns NaN if either group has only one label class.
    
    Args:
        df: DataFrame with scores, labels, and groups
        score_col: Column name for black-box scores
        label_col: Column name for binary labels
        group_col: Column name for group membership
        ref_group: Reference group value (default 0 = white)
        comp_group: Comparison group value (default 1 = black)
        
    Returns:
        ΔAUC value or NaN if computation fails
    """
    df_ref = df[df[group_col] == ref_group]
    df_comp = df[df[group_col] == comp_group]
    
    # Check if groups have both labels
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
# Sampling Utilities
# =============================================================================

def stratified_sampling(n: int, df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    """
    Stratified sampling: sample n rows proportional to group sizes.
    """
    group_counts = df[group_col].value_counts()
    n_per_group = {g: int(np.ceil(n * count / len(df))) for g, count in group_counts.items()}
    
    # Adjust if total exceeds n
    total = sum(n_per_group.values())
    if total > n:
        # Reduce from largest groups
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
    
    return pd.concat(samples, ignore_index=True)


def random_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
) -> pd.DataFrame:
    """Random sampling of k new rows from D not in S."""
    available = D.index.difference(already_sampled_indices)
    if len(available) == 0:
        return pd.DataFrame()
    
    k_actual = min(k, len(available))
    sampled_indices = np.random.choice(available, size=k_actual, replace=False)
    return D.loc[sampled_indices]


def stratified_ordered_sampling(
    k: int,
    S: pd.DataFrame,
    D: pd.DataFrame,
    already_sampled_indices: set,
    group_col: str = "group",
) -> pd.DataFrame:
    """Stratified sampling of k new rows from D not in S."""
    available_df = D.loc[D.index.difference(already_sampled_indices)]
    if len(available_df) == 0:
        return pd.DataFrame()
    
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
    
    Args:
        k: Number of samples to draw
        S: Current sample set (not used, for API consistency)
        D: Full dataset
        already_sampled_indices: Indices already sampled
        bb_scores: Black-box scores for all D rows (precomputed)
        gamma: Power exponent for uncertainty weighting
        
    Returns:
        k newly sampled rows
    """
    available_indices = D.index.difference(already_sampled_indices)
    if len(available_indices) == 0:
        return pd.DataFrame()
    
    # Get scores for available indices
    available_positions = [D.index.get_loc(idx) for idx in available_indices]
    available_scores = bb_scores[available_positions]
    
    # Compute uncertainty proxy: p(1-p)
    p = np.clip(available_scores, 1e-6, 1 - 1e-6)
    uncertainty = p * (1 - p)
    
    # Weight by uncertainty^gamma
    weights = np.power(uncertainty, gamma)
    weights = weights / weights.sum()
    
    # Sample k indices
    k_actual = min(k, len(available_indices))
    sampled_positions = np.random.choice(
        len(available_indices), size=k_actual, replace=False, p=weights
    )
    sampled_indices = available_indices[sampled_positions]
    
    return D.loc[sampled_indices]


# =============================================================================
# Bayesian Optimization Sampling
# =============================================================================

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
    
    Args:
        k: Number of samples to draw
        S: Current sample set
        D: Full dataset
        already_sampled_indices: Indices already sampled
        bb_scores: Black-box scores for all D (precomputed)
        text_embeddings: Embeddings for all D rows
        true_delta: True ΔAUC (for computing objective)
        xi: Exploration parameter for EI
        
    Returns:
        k newly sampled rows
    """
    available_indices = D.index.difference(already_sampled_indices)
    if len(available_indices) == 0:
        return pd.DataFrame()
    
    # Prepare training data from S
    S_indices = [D.index.get_loc(idx) for idx in S.index]
    X_train = text_embeddings[S_indices]
    
    # Compute objective: -|current_delta - true_delta|
    current_delta = compute_delta_auc(S)
    if np.isnan(current_delta):
        current_delta = 0.0
    y_train = np.array([-abs(current_delta - true_delta)] * len(S))
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit GP
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=2)
    
    try:
        gp.fit(X_train_scaled, y_train)
    except Exception:
        # Fallback to random sampling if GP fails
        k_actual = min(k, len(available_indices))
        sampled_indices = np.random.choice(available_indices, size=k_actual, replace=False)
        return D.loc[sampled_indices]
    
    # Predict on available candidates
    available_positions = [D.index.get_loc(idx) for idx in available_indices]
    X_pool = text_embeddings[available_positions]
    X_pool_scaled = scaler.transform(X_pool)
    
    mu, sigma = gp.predict(X_pool_scaled, return_std=True)
    sigma = np.maximum(sigma, 1e-6)
    
    # Expected Improvement
    best_y = y_train.max()
    z = (mu - best_y - xi) / sigma
    ei = (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-6] = 0.0
    
    # Select top k by EI
    k_actual = min(k, len(available_indices))
    top_k_positions = np.argsort(-ei)[:k_actual]
    sampled_indices = available_indices[top_k_positions]
    
    return D.loc[sampled_indices]


# =============================================================================
# Black-box Model Training
# =============================================================================

def train_blackbox_model(
    seed: int,
    device: str = "cuda",
    sbic_path: str = "SBIC_group.csv",
    train_frac: float = 0.9,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 2e-5,
    flip_probs: Optional[Dict[str, float]] = None,
):
    """
    Train a black-box BERT model on SBIC dataset using load_sbic_and_train_api_df.
    
    Args:
        seed: Random seed for training
        device: Device for training
        sbic_path: Path to SBIC CSV file
        train_frac: Fraction of data to use for training
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        flip_probs: Label flip probabilities per group (for biased training)
        
    Returns:
        BlackBoxAPI instance ready for predictions
    """
    # Import BlackBoxAPI here to respect GPU setting
    from blackboxes.blackbox_api_BERT import BlackBoxAPI
    
    # Initialize API
    api = BlackBoxAPI(device=device)
    
    # Set default flip probs if not provided
    if flip_probs is None:
        flip_probs = {"black": 0.9, "white": 0.1}
    
    # Train using load_sbic_and_train_api_df
    print(f"Training black-box model on SBIC (seed={seed}, epochs={epochs})...")
    api = load_sbic_and_train_api_df(
        api=api,
        path=sbic_path,
        flip_probs=flip_probs,
        seed=seed,
        train_frac=train_frac,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    
    return api


def load_jigsaw_data(
    path: str = "jigsaw_group.csv",
    groups: Tuple[str, ...] = ("white", "black"),
    n_sample: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and prepare Jigsaw dataset for auditing using the data_loader function.
    
    Args:
        path: Path to Jigsaw CSV file
        groups: Tuple of group names to include
        n_sample: Number of samples to draw (None to use all 10000 from load_jigsaw)
        seed: Random seed for sampling
        
    Returns:
        DataFrame with columns: text, label, group (int-coded)
    """
    # Load using data_loader function
    df = load_jigsaw(path=path, groups=groups)
    
    # The load_jigsaw function already samples 10000 rows with seed=42
    # If we need a different sample size, resample
    if n_sample is not None and n_sample != len(df):
        if n_sample < len(df):
            df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
        else:
            print(f"Warning: Requested {n_sample} samples but only {len(df)} available")
    
    # Ensure group is int-coded (0/1) with white=0, black=1
    if df["group"].dtype == "object" or df["group"].dtype.name == "category":
        # Explicit mapping: white=0, black=1
        group_map = {"white": 0, "black": 1}
        
        # Check if we have the expected groups
        unique_groups = df["group"].unique()
        for g in unique_groups:
            if g not in group_map:
                print(f"Warning: Unexpected group value '{g}', will be dropped")
        
        # Map groups and drop any unexpected values
        df["group"] = df["group"].map(group_map)
        df = df.dropna(subset=["group"])  # Remove rows with unmapped groups
        df["group"] = df["group"].astype(int)
        
        print(f"Mapped groups: white=0, black=1")
    
    # Validate that we only have 0 and 1
    group_values = df["group"].unique()
    if not set(group_values).issubset({0, 1}):
        raise ValueError(f"Group column must only contain 0 and 1, found: {group_values}")
    
    # Ensure we have 'label' column (might be 'true_label' or 'toxic' in Jigsaw)
    if "label" not in df.columns:
        if "true_label" in df.columns:
            df["label"] = df["true_label"]
            print(f"Created 'label' column from 'true_label'")
        elif "toxic" in df.columns:
            df["label"] = df["toxic"]
            print(f"Created 'label' column from 'toxic'")
    
    # Ensure required columns exist
    required_cols = ["text", "label", "group"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available: {df.columns.tolist()}")
    
    print(f"Loaded Jigsaw dataset: {len(df)} rows")
    print(f"Columns present: {df.columns.tolist()}")
    print(f"Group encoding: white=0, black=1")
    
    # Show group distribution with labels
    group_counts = df['group'].value_counts().sort_index()
    for group_val in group_counts.index:
        count = group_counts[group_val]
        group_name = "white" if group_val == 0 else "black"
        print(f"  Group {group_val} ({group_name}): {count} samples")
    
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


# =============================================================================
# Main Audit Loop
# =============================================================================

def run_single_seed_audit(
    seed: int,
    D: pd.DataFrame,
    api,
    max_queries: int,
    k_init: int,
    k_batch: int,
    strategy: str,
    true_delta: float,
    text_embeddings: Optional[np.ndarray] = None,
    bb_scores_full: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Run audit for a single seed and strategy.
    
    Args:
        seed: Random seed
        D: Full audit dataset
        api: Black-box model API
        max_queries: Maximum query budget
        k_init: Initial stratified sample size
        k_batch: Batch size for ordered sampling
        strategy: Sampling strategy name
        true_delta: True ΔAUC computed on full dataset
        text_embeddings: Precomputed embeddings for BO (optional)
        bb_scores_full: Precomputed BB scores for power/BO (optional)
        
    Returns:
        DataFrame with trajectory: [t_size, delta_auc, abs_error_roc_auc_diff]
    """
    set_all_seeds(seed)
    
    # Initialize with stratified sampling
    S = stratified_sampling(k_init, D)
    S = S.copy()
    S["bb_score"] = api.predict_scores(S["text"].tolist())
    
    already_sampled = set(S.index)
    trajectory = []
    
    # Record initial point
    current_delta = compute_delta_auc(S)
    if not np.isnan(current_delta):
        trajectory.append({
            "t_size": len(S),
            "delta_auc": current_delta,
            "abs_error_roc_auc_diff": abs(current_delta - true_delta),
        })
    
    # Ordered sampling loop
    n_batches = (max_queries - k_init) // k_batch
    
    for batch_idx in tqdm(range(n_batches), desc=f"Seed {seed}, {strategy}", leave=False):
        if len(already_sampled) >= len(D):
            break
        
        # Select next batch based on strategy
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
        
        # Query black-box for new batch
        new_batch = new_batch.copy()
        new_batch["bb_score"] = api.predict_scores(new_batch["text"].tolist())
        
        # Update S and tracking
        S = pd.concat([S, new_batch], ignore_index=True)
        already_sampled.update(new_batch.index)
        
        # Compute current estimate
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
    n_audit: int,
    max_queries: int,
    k_init: int,
    k_batch: int,
    strategies: List[str],
    out_dir: Path,
    device: str = "cuda",
    sbic_path: str = "SBIC_group.csv",
    jigsaw_path: str = "jigsaw_group.csv",
    jigsaw_groups: Tuple[str, ...] = ("white", "black"),
    train_epochs: int = 4,
    train_batch_size: int = 32,
    train_lr: float = 2e-5,
) -> None:
    """
    Run audit across multiple seeds and strategies.
    
    For each seed:
        1. Train fresh black-box model
        2. Load audit dataset and compute true ΔAUC
        3. Run each strategy and save trajectory
    
    Args:
        seeds: List of random seeds
        n_audit: Number of audit samples to load from Jigsaw
        max_queries: Maximum query budget
        k_init: Initial stratified sample size
        k_batch: Batch size for ordered sampling
        strategies: List of strategy names
        out_dir: Output directory for results
        device: Device for model training
        sbic_path: Path to SBIC CSV
        jigsaw_path: Path to Jigsaw CSV
        jigsaw_groups: Tuple of group names to include
        train_epochs: Training epochs for black-box
        train_batch_size: Training batch size
        train_lr: Training learning rate
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all trajectories
    all_trajectories = {strategy: [] for strategy in strategies}
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx+1}/{len(seeds)})")
        print(f"{'='*60}")
        
        set_all_seeds(seed)
        
        # 1. Train black-box model
        print(f"Training black-box model for seed {seed}...")
        api = train_blackbox_model(
            seed=seed,
            device=device,
            sbic_path=sbic_path,
            epochs=train_epochs,
            batch_size=train_batch_size,
            lr=train_lr,
        )
        
        # 2. Load audit dataset
        print(f"Loading Jigsaw audit dataset...")
        D = load_jigsaw_data(
            path=jigsaw_path,
            groups=jigsaw_groups,
            n_sample=n_audit,
            seed=seed,
        )
        
        # Compute true ΔAUC on full dataset
        print("Computing true ΔAUC on full audit set...")
        D_full = D.copy()
        D_full["bb_score"] = api.predict_scores(D_full["text"].tolist())
        true_delta = compute_delta_auc(D_full)
        print(f"True ΔAUC = {true_delta:.4f}")
        
        # 3. Precompute embeddings and scores for efficiency
        # For power and BO strategies, precompute BB scores once
        bb_scores_full = None
        text_embeddings = None
        
        if "power" in strategies or "bo" in strategies:
            print("Precomputing black-box scores for full dataset...")
            bb_scores_full = api.predict_scores(D["text"].tolist())
        
        if "bo" in strategies:
            print("Computing text embeddings for BO...")
            # Simple TF-IDF embeddings (replace with better embeddings if needed)
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
            text_embeddings = vectorizer.fit_transform(D["text"]).toarray()
        
        # 4. Run each strategy
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
            
            # Add metadata
            trajectory_df["seed"] = seed
            trajectory_df["strategy"] = strategy
            trajectory_df["true_delta"] = true_delta
            
            # Save per-seed trajectory
            seed_file = out_dir / f"trajectory_{strategy}_seed{seed}.csv"
            trajectory_df.to_csv(seed_file, index=False)
            print(f"Saved: {seed_file}")
            
            # Accumulate for combined output
            all_trajectories[strategy].append(trajectory_df)
    
    # 5. Combine and save all trajectories
    print(f"\n{'='*60}")
    print("Saving combined trajectories...")
    for strategy, traj_list in all_trajectories.items():
        combined_df = pd.concat(traj_list, ignore_index=True)
        combined_file = out_dir / f"trajectory_{strategy}_all.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"Saved: {combined_file}")


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_summary_statistics(
    out_dir: Path,
    strategies: List[str],
    eps_thresholds: List[float],
    max_queries: int,
    target_query_budget: int = 250,
) -> pd.DataFrame:
    """
    Compute summary statistics across all seeds for each strategy.
    
    Includes two types of convergence metrics:
    1. mean_crossing_T: When does the MEAN trajectory cross epsilon? (population-level)
    2. queries_to_eps: Average of when individual seeds cross epsilon (per-seed)
    
    Metrics:
        - mean_crossing_T: Query budget when mean(error) first drops below epsilon
        - error_at_crossing_T: Error statistics at mean crossing time
        - queries_to_{eps}_mean/std/n_reached: Per-seed crossing times (mean/std/count)
        - auec_mean/std: Normalized area under error curve
        - error_at_{budget}_mean/std: Interpolated error at target query budget
    
    Args:
        out_dir: Directory containing trajectory CSVs
        strategies: List of strategy names
        eps_thresholds: Error thresholds (e.g., [0.02, 0.05])
        max_queries: Maximum query budget
        target_query_budget: Query budget for interpolated error
        
    Returns:
        Summary DataFrame with aggregated metrics
    """
    summary_rows = []
    
    for strategy in strategies:
        combined_file = out_dir / f"trajectory_{strategy}_all.csv"
        if not combined_file.exists():
            print(f"Warning: {combined_file} not found, skipping {strategy}")
            continue
        
        df_all = pd.read_csv(combined_file)
        seeds = df_all["seed"].unique()
        
        # Compute mean trajectory (aggregate across seeds first)
        summary_df = (
            df_all.groupby("t_size")["abs_error_roc_auc_diff"]
            .agg(["mean", "std", "count", "median"])
            .reset_index()
        )
        
        # Per-epsilon metrics
        for eps in eps_thresholds:
            # ===== Method 1: Mean crossing T (population-level) =====
            # When does the MEAN trajectory cross epsilon?
            crossed = summary_df[summary_df["mean"] <= eps]
            mean_crossing_T = float(crossed["t_size"].min()) if len(crossed) > 0 else np.nan
            
            # Error stats at mean crossing time
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
            
            # ===== Method 2: Per-seed crossing times (existing approach) =====
            # When does each individual seed cross epsilon?
            seed_crossing_times = []
            for seed in seeds:
                df_seed = df_all[df_all["seed"] == seed].copy()
                df_seed = df_seed.sort_values("t_size")
                
                below_eps = df_seed[df_seed["abs_error_roc_auc_diff"] <= eps]
                if len(below_eps) > 0:
                    seed_crossing_times.append(float(below_eps["t_size"].iloc[0]))
                else:
                    seed_crossing_times.append(np.nan)
            
            # Aggregate per-seed crossing times
            valid_crossings = [t for t in seed_crossing_times if not np.isnan(t)]
            queries_to_eps_mean = np.mean(valid_crossings) if valid_crossings else np.nan
            queries_to_eps_std = np.std(valid_crossings) if valid_crossings else np.nan
            queries_to_eps_median = np.median(valid_crossings) if valid_crossings else np.nan
            n_converged = len(valid_crossings)
        
            # Build row for this strategy-epsilon combination
            row = {
                "strategy": strategy,
                "epsilon": eps,
                # Population-level crossing (mean trajectory)
                "mean_crossing_T": mean_crossing_T,
                "error_at_crossing_T_mean": error_at_crossing_mean,
                "error_at_crossing_T_std": error_at_crossing_std,
                "error_at_crossing_T_median": error_at_crossing_median,
                "n_at_crossing_T": n_at_crossing,
                # Per-seed crossing times
                f"queries_to_{eps}_mean": queries_to_eps_mean,
                f"queries_to_{eps}_std": queries_to_eps_std,
                f"queries_to_{eps}_median": queries_to_eps_median,
                f"queries_to_{eps}_n_reached": n_converged,
            }
            
            summary_rows.append(row)
        
        # ===== Compute AUEC and error_at_target once per strategy =====
        # (These don't depend on epsilon)
        auec_values = []
        error_at_target = []
        
        for seed in seeds:
            df_seed = df_all[df_all["seed"] == seed].copy()
            df_seed = df_seed.sort_values("t_size")
            
            # AUEC
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
                print(f"Warning: AUEC computation failed for {strategy} seed {seed}: {e}")
                auec_values.append(np.nan)
            
            # Error at target budget (linear interpolation)
            if target_query_budget in df_seed["t_size"].values:
                error_at_target.append(
                    df_seed[df_seed["t_size"] == target_query_budget]["abs_error_roc_auc_diff"].iloc[0]
                )
            else:
                # Linear interpolation
                df_sorted = df_seed.sort_values("t_size")
                before = df_sorted[df_sorted["t_size"] <= target_query_budget]
                after = df_sorted[df_sorted["t_size"] >= target_query_budget]
                
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
        
        # Add AUEC and error_at_target to all rows for this strategy
        auec_clean = [v for v in auec_values if not np.isnan(v)]
        auec_mean = np.mean(auec_clean) if auec_clean else np.nan
        auec_std = np.std(auec_clean) if auec_clean else np.nan
        
        error_clean = [v for v in error_at_target if not np.isnan(v)]
        error_at_target_mean = np.mean(error_clean) if error_clean else np.nan
        error_at_target_std = np.std(error_clean) if error_clean else np.nan
        
        # Add to all rows for this strategy
        for row in summary_rows:
            if row["strategy"] == strategy:
                row["auec_mean"] = auec_mean
                row["auec_std"] = auec_std
                row[f"error_at_{target_query_budget}_mean"] = error_at_target_mean
                row[f"error_at_{target_query_budget}_std"] = error_at_target_std
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Reorder columns for better readability
    col_order = ["strategy", "epsilon"]
    # Add mean_crossing_T columns first
    col_order.extend([c for c in summary_df.columns if "crossing_T" in c and c not in col_order])
    # Then queries_to_eps columns
    col_order.extend([c for c in summary_df.columns if "queries_to" in c and c not in col_order])
    # Then other metrics
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit baselines for Jigsaw dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (None for CPU)",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=2000,
        help="Maximum query budget",
    )
    parser.add_argument(
        "--k_init",
        type=int,
        default=4,
        help="Initial stratified sample size",
    )
    parser.add_argument(
        "--k_batch",
        type=int,
        default=16,
        help="Batch size for ordered sampling",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="0.02,0.05",
        help="Comma-separated error thresholds",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,stratified,power,bo",
        help="Comma-separated list of strategies (random, stratified, power, bo)",
    )
    parser.add_argument(
        "--n_audit",
        type=int,
        default=10000,
        help="Number of audit samples to load from Jigsaw",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="evaluation/baselines/results_jigsaw",
        help="Output directory for results",
    )
    parser.add_argument(
        "--target_query_budget",
        type=int,
        default=250,
        help="Query budget for interpolated error metric",
    )
    parser.add_argument(
        "--sbic_path",
        type=str,
        default="SBIC_group.csv",
        help="Path to SBIC CSV file",
    )
    parser.add_argument(
        "--jigsaw_path",
        type=str,
        default="jigsaw_group.csv",
        help="Path to Jigsaw CSV file",
    )
    parser.add_argument(
        "--jigsaw_groups",
        type=str,
        default="white,black",
        help="Comma-separated group names to include from Jigsaw",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Number of epochs for black-box training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for black-box training",
    )
    parser.add_argument(
        "--train_lr",
        type=float,
        default=2e-5,
        help="Learning rate for black-box training",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set GPU before importing torch-heavy modules
    set_gpu(args.gpu)
    
    # Parse arguments
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    eps_thresholds = [float(e.strip()) for e in args.eps.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    jigsaw_groups = tuple(g.strip() for g in args.jigsaw_groups.split(","))
    out_dir = Path(args.out_dir)
    
    print("=" * 60)
    print("Audit Baselines - Jigsaw Dataset")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    print(f"Strategies: {strategies}")
    print(f"Max queries: {args.max_queries}")
    print(f"Initial sample size: {args.k_init}")
    print(f"Batch size: {args.k_batch}")
    print(f"Audit dataset size: {args.n_audit}")
    print(f"Error thresholds: {eps_thresholds}")
    print(f"SBIC path: {args.sbic_path}")
    print(f"Jigsaw path: {args.jigsaw_path}")
    print(f"Jigsaw groups: {jigsaw_groups}")
    print(f"Training epochs: {args.train_epochs}")
    print(f"Training batch size: {args.train_batch_size}")
    print(f"Training LR: {args.train_lr}")
    print(f"Output directory: {out_dir}")
    print(f"GPU: {args.gpu if args.gpu is not None else 'CPU'}")
    print("=" * 60)
    
    # Device selection
    import torch
    device = "cuda" if torch.cuda.is_available() and args.gpu is not None else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run audit
    start_time = time.time()
    
    run_multi_seed_audit(
        seeds=seeds,
        n_audit=args.n_audit,
        max_queries=args.max_queries,
        k_init=args.k_init,
        k_batch=args.k_batch,
        strategies=strategies,
        out_dir=out_dir,
        device=device,
        sbic_path=args.sbic_path,
        jigsaw_path=args.jigsaw_path,
        jigsaw_groups=jigsaw_groups,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
    )
    
    # Compute summary statistics
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
