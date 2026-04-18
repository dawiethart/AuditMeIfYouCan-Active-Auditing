# evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import wandb
from utils import stratified_sampling

def compute_blackbox_auc_difference(
    labels: pd.Series, groups: pd.Series, scores: np.ndarray, group1="white", group2="black"
) -> float:
    """
    Compute the difference in AUC between two groups using model scores.

    Parameters:
        labels (pd.Series): True binary labels (0/1).
        groups (pd.Series): Group membership for each instance.
        scores (np.ndarray): Predicted probabilities or scores from black-box.
        group1 (str): Name of group 1 (e.g., 'white').
        group2 (str): Name of group 2 (e.g., 'black').

    Returns:
        float: AUC(group1) - AUC(group2), or 0 if one group fails.
    """
    df = pd.DataFrame({"score": scores, "true_label": labels, "group": groups})

    try:
        auc1 = roc_auc_score(
            df[df["group"] == group1]["true_label"], df[df["group"] == group1]["score"]
        )
    except ValueError:
        auc1 = 0.0

    try:
        auc2 = roc_auc_score(
            df[df["group"] == group2]["true_label"], df[df["group"] == group2]["score"]
        )
    except ValueError:
        auc2 = 0.0
    
    return abs(auc1 - auc2)

# evaluation.py

def evaluate_outer_loop(
    ground_truth_delta: float,
    D: pd.DataFrame,
    sample_S: pd.DataFrame,
    sample_scores: np.ndarray,
    random_S: pd.DataFrame,
    random_scores: np.ndarray,
    strat_S: pd.DataFrame,
    stratified_scores: np.ndarray,
    true_label_col: str = "true_label",
    group_col: str = "group",
    group1: str = "white",
    group2: str = "black",
) -> dict:
    """
    Logs ΔAUC across:
      • true full-D black-box
      • your queried sample S
      • random baseline of same size
      • stratified baseline of same size
    """
    def delta_auc(df, scores):
        lbl = df[true_label_col].astype(int).values
        grp = df[group_col].values
        return abs(
            roc_auc_score(lbl[grp == group1], scores[grp == group1])
          - roc_auc_score(lbl[grp == group2], scores[grp == group2])
        )

    auc_samp  = delta_auc(sample_S,    sample_scores)
    auc_rand  = delta_auc(random_S,    random_scores)
    auc_strat = delta_auc(strat_S,     stratified_scores)

    wandb.log({
        "ΔAUC/true":       float(ground_truth_delta),
        "ΔAUC/sample":     float(auc_samp),
        "ΔAUC/random":     float(auc_rand),
        "ΔAUC/stratified": float(auc_strat),

        "ΔAUC/abs_error/sample":     abs(auc_samp  - ground_truth_delta),
        "ΔAUC/abs_error/random":     abs(auc_rand  - ground_truth_delta),
        "ΔAUC/abs_error/stratified": abs(auc_strat - ground_truth_delta),
    })

    return {
        "true":       ground_truth_delta,
        "sample":     auc_samp,
        "random":     auc_rand,
        "stratified": auc_strat,
    }


def evaluate_inner_loop(
    D: pd.DataFrame,
    weights_history: list,        # list of pd.Series of current weights over D
    thresholds: pd.Series,        # current threshold per index
    delta1: np.ndarray,        # current h1(X)
    delta2: np.ndarray,        # current h2(X)
    
    epsilon: float,
):
    """
    For each inner iteration:
      • compute ΔAUC between h1 and h2 and log it
      • log weight distribution stats (min/median/max)
      • detect which IDs exceed thresholds this round
      • log these IDs and their groups/texts
    """
    # ΔAUC between h1 and h2
    delta_auc = abs(delta1-delta2)
    print(f"inner/ΔAUC(h1,h2):{delta_auc}")
    wandb.log({"inner/ΔAUC(h1,h2)": float(delta_auc)})

    # weights stats
    curr_w = weights_history[-1]
    wandb.log({
        "inner/weight_min": float(curr_w.min()),
        "inner/weight_median": float(curr_w.median()),
        "inner/weight_max": float(curr_w.max()),
    })

    # which points exceed threshold?
    over = curr_w >= thresholds
    ids_over = curr_w.index[over]
    wandb.log({"inner/num_exceed": int(over.sum())})

    # log the actual IDs, groups, and optionally texts
    # be careful: logging a large list can overflow
    wandb.log({
        "inner/ids_exceed": wandb.Table(
            data=D.loc[ids_over, ["id", "group", "text"]].values.tolist(),
            columns=["id", "group", "text"]
        )
    })

    # log when ΔAUC below tolerance
    if delta_auc <= 2 * epsilon:
        wandb.log({"inner/converged": True})
    else:
        wandb.log({"inner/converged": False})

    return delta_auc

