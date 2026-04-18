# utils.py
from torchmetrics.functional import auroc
import numpy as np
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
import os
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
import torch
from typing import Callable, Dict, Any, Tuple, List
import gc # to empty the GPU memory

def df_map(dataset, tokenizer, surrogate):
    df = dataset.copy()

    # Assign labels depending on surrogate mode
    df["labels"] = df["bb_score"] if surrogate else df["true_label"]

    # === Convert group labels (e.g. "male", "female") to integers ===
    if "group" in df.columns:
        unique_groups = sorted(df["group"].unique())
        group_map = {g: i for i, g in enumerate(unique_groups)}
        df["group"] = df["group"].map(group_map)
        print(f"[INFO] Group mapping used: {group_map}")
    else:
        print("[WARNING] 'group' column not found in dataset.")

    df_mapped = Dataset.from_pandas(df)
    df_mapped = df_mapped.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    return df, df_mapped

@torch.no_grad()
def compute_group_auc_difference_from_scores(
    scores,
    dataset: pd.DataFrame,
    group1=1,
    group2=0,
    device="cpu",
) -> Tuple[float, torch.Tensor]:
    """
    Compute *unweighted* AUC difference between two groups given prediction scores.

    ΔAUC = AUC(group2) − AUC(group1)
    """
    # Sanitize inputs
    probs = torch.as_tensor(np.asarray(scores, dtype=np.float32), device=device).view(-1)

    # Convert labels to int64
    if not np.issubdtype(dataset["true_label"].dtype, np.number):
        labels_np = pd.to_numeric(dataset["true_label"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        labels_np = dataset["true_label"].to_numpy(dtype=np.int64)
    labels = torch.as_tensor(labels_np, device=device, dtype=torch.long).view(-1)

    # Convert groups to int codes
    gcol = dataset["group"]
    if not np.issubdtype(gcol.dtype, np.number):
        uniq, inv = np.unique(gcol.astype(str).to_numpy(), return_inverse=True)
        groups_np = inv.astype(np.int64)
    else:
        uniq = None
        groups_np = gcol.to_numpy(dtype=np.int64)
    groups = torch.as_tensor(groups_np, device=device, dtype=torch.long).view(-1)

    # Length check
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

    # Create group masks
    mask1 = (groups == int(group1))
    mask2 = (groups == int(group2))

    def safe_auroc(p, y):
        return auroc(p, y, task="binary") if y.unique().numel() >= 2 else torch.tensor(float("nan"), device=device)

    auc1 = safe_auroc(probs[mask1], labels[mask1])
    auc2 = safe_auroc(probs[mask2], labels[mask2])

    return (auc2 - auc1).item(), probs.cpu()



def tokenize_batch(batch, tokenizer):
    tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    return tokenized

def fresh_model():
    """Create a fresh BERT model for sequence classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1
    )

def free_model(model):
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()

def stratified_sampling(size: int, dataset: pd.DataFrame, with_replacement: bool=False) -> pd.DataFrame:
    """
    Strict stratification over (group, true_label) aiming for equal counts per cell.
    Falls back gracefully if some cells are tiny or empty.

    Behavior:
      - Compute active (non-empty) strata over (group, true_label).
      - Allocate floor(size / #active_strata) to each cell, then distribute the remainder
        one-by-one across cells with remaining capacity.
      - If size < #active_strata, pick 1 from as many strata as size.
      - Without replacement: never sample more than available in a cell.
      - With replacement: a cell can supply more than it has.
    """
    df = dataset.copy()
    assert "group" in df.columns and "true_label" in df.columns, "Missing required columns."

    # Identify strata
    groups = sorted(df["group"].unique().tolist())
    labels = sorted(df["true_label"].unique().tolist())
    strata = []
    for g in groups:
        for y in labels:
            idx = df.index[(df["group"] == g) & (df["true_label"] == y)]
            if len(idx) > 0:
                strata.append(((g, y), idx))

    if len(strata) == 0 or size <= 0:
        print("[Init S] No non-empty strata or non-positive size. Returning empty.")
        return df.iloc[0:0].copy()

    n_cells = len(strata)

    # Helper: sample k indices from a given index list
    def sample_indices(idx, k):
        if with_replacement:
            # allow repeats if k > len(idx)
            return np.random.choice(idx, size=k, replace=True).tolist()
        else:
            k = min(k, len(idx))
            if k <= 0:
                return []
            return np.random.choice(idx, size=k, replace=False).tolist()

    chosen = []

    # Case: very small size -> pick 1 from as many strata as we can
    if size < n_cells:
        # round-robin 1 each until we hit 'size'
        for _, idx in strata[:size]:
            chosen.extend(sample_indices(idx, 1))
        out = df.loc[chosen].reset_index(drop=True)
        ct = pd.crosstab(out["group"], out["true_label"])
        print("[Init S] contingency\n", ct)
        return out

    # Base equal allocation
    base = size // n_cells
    remainder = size % n_cells

    # Track remaining capacity per stratum (for no-replacement case)
    # For replacement, capacity is effectively infinite
    capacities = []
    for (_, idx) in strata:
        cap = float("inf") if with_replacement else len(idx)
        capacities.append(cap)

    # First pass: give each stratum 'base'
    for i, ((_, idx)) in enumerate(strata):
        k = base if with_replacement else min(base, capacities[i])
        if k > 0:
            picks = sample_indices(idx, k)
            chosen.extend(picks)
            if not with_replacement:
                capacities[i] -= len(picks)
                # remove picked indices from idx to avoid repeats later
                strata[i] = (strata[i][0], idx.difference(picks))

    # Distribute the remainder 1-by-1 to cells with capacity left
    while remainder > 0:
        # pick the next stratum that can still supply one more
        assigned = False
        for i, ((_, idx)) in enumerate(strata):
            if with_replacement or capacities[i] > 0:
                picks = sample_indices(idx, 1)
                if len(picks) > 0:
                    chosen.extend(picks)
                    remainder -= 1
                    assigned = True
                    if not with_replacement:
                        capacities[i] -= 1
                        strata[i] = (strata[i][0], idx.difference(picks))
                if remainder == 0:
                    break
        if not assigned:
            # No cell can provide more (no-replacement & all exhausted). Stop.
            break

    # If we still didn't reach 'size' due to exhaustion (no-replacement), fill globally
    if len(chosen) < size:
        short = size - len(chosen)
        already = set(chosen)
        pool = df.index.difference(already)
        if len(pool) > 0:
            extra = np.random.choice(pool, size=min(short, len(pool)), replace=False).tolist()
            chosen.extend(extra)
        elif with_replacement:
            # last resort: sample with replacement from entire df
            extra = np.random.choice(df.index, size=short, replace=True).tolist()
            chosen.extend(extra)

    out = df.loc[chosen[:size]].reset_index(drop=True)
    ct = pd.crosstab(out["group"], out["true_label"])
    print("[Init S] contingency\n", ct)
    return out




def select_topk_stratified_disagreement(df_D, disagreements, top_k_per_bucket=5):
    """
    Selects top-k disagreement examples per group-label bucket from df_D.

    Parameters:
        df_D (pd.DataFrame): Full dataset D with at least columns ['group', 'true_label', 'text', 'id']
        disagreements (np.array): Array of disagreement values, same length as df_D
        top_k_per_bucket (int): Number of examples to select per group-label bucket

    Returns:
        new_T (pd.DataFrame): Selected top-k*4 examples with high disagreement, stratified by group and label
    """
    df_eval = df_D.copy()
    df_eval["disagreement"] = disagreements

    # Define stratified buckets
    buckets = {
        "white_1": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 1)],
        "white_0": df_eval[(df_eval["group"] == 1) & (df_eval["true_label"] == 0)],
        "black_1": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 1)],
        "black_0": df_eval[(df_eval["group"] == 0) & (df_eval["true_label"] == 0)],
    }

    selected_dfs = []
    for key, bucket in buckets.items():
        sorted_bucket = bucket.sort_values("disagreement", ascending=False)
        top_k = sorted_bucket.head(top_k_per_bucket)
        selected_dfs.append(top_k)

    new_T = pd.concat(selected_dfs).drop_duplicates(subset="id")
    return new_T


def delta_progress(df_new, df_old, iteration, delta_auc_blackbox, compute_auc_fn):
    """
    Computes the progression of delta AUC as more points are added.

    Parameters:
        df_new (pd.DataFrame): New evaluation samples.
        df_old (pd.DataFrame): Previously evaluated samples.
        iteration (int): Current iteration.
        delta_auc_blackbox (float): Baseline delta AUC to compare against.
        compute_auc_fn (Callable): Function to compute group AUC difference.

    Returns:
        List[float]: List of delta values at each prefix of df_new.
    """
    delta_auc = []

    for i in range(len(df_new)):
        if iteration == 0:
            df_progress = df_new.iloc[:i]
        else:
            df_progress = pd.concat([df_new.iloc[:i], df_old])

        delta = compute_auc_fn(
            labels=df_progress["true_label"].astype(int),
            groups=df_progress["group"],
            scores=df_progress["bb_score"],
        )
        delta_auc.append(abs(delta - delta_auc_blackbox))

    return delta_auc



def random_ordered_sampling(D: pd.DataFrame, api_fn, seed: int = None):
    """
    Return:
      • rand_D: DataFrame with all rows of D in a random permutation
      • scores:  np.ndarray of black-box scores in that same permuted order
    """
    # shuffle indices without replacement
    rng = np.random.default_rng(seed)
    perm = rng.permutation(D.index.values)
    rand_D = D.loc[perm].reset_index(drop=True)
    # query black‐box once for each text, in this new order
    scores = np.array(api_fn(rand_D["text"].tolist()))
    return rand_D, scores


def create_stratified_batches(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """
    Return a *reordered* dataframe where rows are interleaved across
    the four (group, true_label) strata as much as possible.

    The 'batch_size' argument is only used for logging; the actual
    batching is handled by the DataLoader. The idea is that with this
    ordering, every contiguous block of `batch_size` rows will be
    approximately balanced across strata.
    """
    if "group" not in df.columns or "true_label" not in df.columns:
        raise ValueError("create_stratified_batches expects 'group' and 'true_label' columns.")

    df = df.copy().reset_index(drop=True)

    # We assume binary group {0,1} and label {0,1} for now.
    # If a stratum is empty, we just skip it.
    strata = {}
    for g in sorted(df["group"].unique()):
        for y in sorted(df["true_label"].unique()):
            mask = (df["group"] == g) & (df["true_label"] == y)
            if mask.any():
                # shuffle within each stratum
                strata[(g, y)] = df[mask].sample(frac=1.0, random_state=42).reset_index(drop=True)

    if not strata:
        # nothing special to do
        return df

    # Round-robin interleaving across available strata
    keys = list(strata.keys())
    positions = {k: 0 for k in keys}
    total_len = sum(len(s) for s in strata.values())

    rows = []
    while len(rows) < total_len:
        progress = False
        for k in keys:
            pos = positions[k]
            s = strata[k]
            if pos < len(s):
                rows.append(s.iloc[pos])
                positions[k] += 1
                progress = True
            # if a stratum is exhausted, we just skip it
        if not progress:
            break  # all strata exhausted

    strat_df = pd.DataFrame(rows).reset_index(drop=True)

    print(
        f"[create_stratified_batches] Reordered {len(df)} rows into stratified sequence "
        f"for batch_size={batch_size}."
    )
    print("[create_stratified_batches] contingency after reordering:\n",
          pd.crosstab(strat_df["group"], strat_df["true_label"]))

    return strat_df


def stratified_ordered_sampling(D, api_fn, group_col='group', label_col='labels', 
                                group1=0, group2=1, seed=None):
    """
    Stratified sampling by both group and label.
    Ensures balanced representation across all four strata:
    - group1 × positive
    - group1 × negative  
    - group2 × positive
    - group2 × negative
    """
    rng = np.random.default_rng(seed)
    
    # Create four strata
    g1_pos = D[(D[group_col] == group1) & (D[label_col] == 1)].sample(
        frac=1, random_state=int(rng.integers(1e9))
    ).reset_index(drop=True)
    
    g1_neg = D[(D[group_col] == group1) & (D[label_col] == 0)].sample(
        frac=1, random_state=int(rng.integers(1e9))
    ).reset_index(drop=True)
    
    g2_pos = D[(D[group_col] == group2) & (D[label_col] == 1)].sample(
        frac=1, random_state=int(rng.integers(1e9))
    ).reset_index(drop=True)
    
    g2_neg = D[(D[group_col] == group2) & (D[label_col] == 0)].sample(
        frac=1, random_state=int(rng.integers(1e9))
    ).reset_index(drop=True)
    
    # Find minimum count across all four strata
    min_n = min(len(g1_pos), len(g1_neg), len(g2_pos), len(g2_neg))
    
    # Interleave all four strata in a round-robin fashion
    interleaved_rows = []
    for i in range(min_n):
        interleaved_rows.append(g1_pos.iloc[i])
        interleaved_rows.append(g1_neg.iloc[i])
        interleaved_rows.append(g2_pos.iloc[i])
        interleaved_rows.append(g2_neg.iloc[i])
    
    interleaved = pd.DataFrame(interleaved_rows).reset_index(drop=True)
    
    # Add remainders from all strata
    remainder = pd.concat([
        g1_pos.iloc[min_n:],
        g1_neg.iloc[min_n:],
        g2_pos.iloc[min_n:],
        g2_neg.iloc[min_n:]
    ]).reset_index(drop=True)
    
    strat_D = pd.concat([interleaved, remainder]).reset_index(drop=True)
    
    scores = np.array(api_fn(strat_D["text"].tolist()))
    return strat_D, scores


def plot_weight_evolution(weight_history, selected_ids_history, save_dir="audit_plots"):
    """
    Plots and saves:
    1. Weight evolution of tracked sample IDs across iterations.
    2. Size of T (number of selected samples) per iteration.

    Parameters:
        weight_history: List of pd.Series (sample weights at each iteration)
        selected_ids_history: List of pd.Series (selected IDs at each iteration)
        save_dir: Directory where to save the plots (default: 'audit_plots')
    """
    os.makedirs(save_dir, exist_ok=True)
    iterations = list(range(len(weight_history)))

    # === Plot 1: Weight evolution for tracked example IDs ===
    all_ids = set().union(*[set(w.index) for w in weight_history])
    tracked_ids = list(all_ids)[:10]

    selected_indices = list(range(0, len(weight_history), 1))[:6]
    num_plots = len(selected_indices)

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), sharey=True)
    if num_plots == 1:
        axes = [axes]

    for ax, i in zip(axes, selected_indices):
        weights = weight_history[i]
        ax.hist(weights, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(f"Iteration {i}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.grid(True)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "weight_evolution.png")
    plt.savefig(path1)
    plt.show()

    # === Plot 2: Number of selected T samples per iteration ===
    plt.figure(figsize=(8, 4))
    t_sizes = [len(s) for s in selected_ids_history]
    plt.plot(iterations, t_sizes, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of T samples added")
    plt.title("T Size Over Iterations")
    plt.grid(True)
    plt.tight_layout()
    path2 = os.path.join(save_dir, "t_size_over_iterations.png")
    plt.savefig(path2)
    plt.show()

    print(f"Plots saved to:\n - {path1}\n - {path2}")


import numpy as np
import pandas as pd


def sample_stratified_fixed_size(
    df: pd.DataFrame,
    n: int,
    rng: np.random.RandomState,
    group_col: str = "group",
    label_col: str = "true_label",
) -> pd.DataFrame:
    """
    Proportional stratified sampling over (group, label) to obtain exactly n rows.

    Each stratum (g, y) gets approximately:
        alloc(g,y) ≈ (N_{g,y} / N) * n
    using floor + "largest fractional remainder" allocation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain group_col and label_col.
    n : int
        Desired total sample size (n <= len(df) recommended).
    rng : np.random.RandomState
        Random state for reproducibility.
    group_col : str
        Column name for group membership.
    label_col : str
        Column name for true labels.

    Returns
    -------
    pd.DataFrame
        Stratified subsample of df of length n (shuffled).
    """
    if n >= len(df):
        # Just return a shuffled copy of the entire df
        return df.sample(frac=1.0, replace=False, random_state=rng).reset_index(drop=True)

    # Group by (group, label)
    grouped = df.groupby([group_col, label_col], dropna=False)

    # Population counts and ratios
    pop_counts = grouped.size()
    N = float(len(df))
    pop_ratios = pop_counts / N  # ratio per stratum

    # Ideal (float) allocation
    alloc_float = pop_ratios * float(n)
    alloc_floor = np.floor(alloc_float).astype(int)

    # How many have we allocated with the floors?
    allocated = int(alloc_floor.sum())
    leftover = int(n - allocated)

    # Fractional parts for largest-remainder allocation
    frac = alloc_float - alloc_floor

    # Initialize extra allocation with zeros
    extra = np.zeros_like(alloc_floor, dtype=int)

    if leftover > 0:
        # Indices of strata sorted by descending fractional part
        frac_sorted_idx = np.argsort(-frac.values)
        for idx in frac_sorted_idx[:leftover]:
            extra[idx] += 1

    # Final allocation per stratum
    alloc = alloc_floor + extra
    assert alloc.sum() == n, f"Allocation sum {alloc.sum()} != n {n}"

    # Draw samples in each stratum according to alloc
    sampled_parts = []
    for (key, group_df), k in zip(grouped, alloc):
        k = int(k)
        if k <= 0:
            continue
        # n <= len(df) plus proportional allocation guarantees k <= len(group_df)
        sampled_group = group_df.sample(n=k, replace=False, random_state=rng)
        sampled_parts.append(sampled_group)

    if not sampled_parts:
        raise RuntimeError("No samples drawn in sample_stratified_fixed_size; check inputs.")

    sampled = pd.concat(sampled_parts, axis=0)
    # Shuffle the concatenated sample
    sampled = sampled.sample(frac=1.0, replace=False, random_state=rng).reset_index(drop=True)
    return sampled
