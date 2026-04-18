#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# Strategy exclusion helpers (NEW)
# -----------------------------
def _parse_exclude_list(s: str) -> List[str]:
    """
    Parse comma-separated list into lowercase tokens.
    Examples:
      "" -> []
      "random, baseline:power ,bafa:bo" -> ["random","baseline:power","bafa:bo"]
    """
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _apply_strategy_excludes(
    df: pd.DataFrame,
    *,
    strategy_col: str,
    exclude: List[str],
    source_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Exclude strategies from a dataframe.

    exclude matches:
      - raw strategy names in df[strategy_col] (case-insensitive), e.g. "random"
      - if source_prefix is provided, also matches "source:strategy", e.g. "baseline:random"
        (useful when filtering raw dataframes pre-prefixing)

    If df doesn't have strategy_col or exclude is empty, returns df unchanged.
    """
    if not exclude:
        return df
    if strategy_col not in df.columns:
        return df

    ex = set(exclude)
    s_raw = df[strategy_col].astype(str).str.lower()

    # match raw
    mask = ~s_raw.isin(ex)

    # optionally match "source:raw"
    if source_prefix is not None:
        s_pref = (source_prefix.lower() + ":" + s_raw)
        mask &= ~s_pref.isin(ex)

    return df.loc[mask].copy()


# -----------------------------
# Statistical helpers (NEW)
# -----------------------------
def _compute_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for an array of values.
    Returns (lower_bound, upper_bound).
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    
    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return float(mean - ci), float(mean + ci)


def _descriptive_stats(values: np.ndarray) -> dict:
    """
    Compute comprehensive descriptive statistics for a set of values.
    Returns dict with mean, std, median, IQR, min, max, n, ci_low, ci_high.
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n": 0,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    
    ci_low, ci_high = _compute_ci(values)
    q25, q75 = np.percentile(values, [25, 75])
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# -----------------------------
# Aggregation helpers (unchanged)
# -----------------------------
def _ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {list(df.columns)}")


def _prep_group_series(
    g: pd.DataFrame, t_col: str, err_col: str, t_start: int, t_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    d = g[[t_col, err_col]].dropna().copy()
    if d.empty:
        return np.array([]), np.array([])
    d[t_col] = pd.to_numeric(d[t_col], errors="coerce")
    d[err_col] = pd.to_numeric(d[err_col], errors="coerce")
    d = d.dropna(subset=[t_col, err_col])

    d = d.sort_values(t_col).drop_duplicates(subset=[t_col], keep="last")
    d = d[(d[t_col] >= t_start) & (d[t_col] <= t_max)]
    if d.empty:
        return np.array([]), np.array([])
    return d[t_col].to_numpy(dtype=float), d[err_col].to_numpy(dtype=float)


def _interp_to_grid(t: np.ndarray, e: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, t, e, left=float(e[0]), right=float(e[-1]))


def aggregate_curves(
    df: pd.DataFrame,
    *,
    source_name: str,
    t_col: str,
    err_col: str,
    strategy_col: str,
    replicate_cols: List[str],
    t_start: int,
    t_max: int,
    grid_step: int,
    ci_z: float = 1.96,
) -> pd.DataFrame:
    _ensure_cols(df, [t_col, err_col, strategy_col] + replicate_cols, source_name)

    grid = np.arange(t_start, t_max + 1, grid_step, dtype=float)

    out_rows = []
    for strategy, sdf in df.groupby(strategy_col, dropna=False):
        grp = sdf.groupby(replicate_cols, dropna=False)

        curves = []
        for _, g in grp:
            t, e = _prep_group_series(g, t_col, err_col, t_start=t_start, t_max=t_max)
            y = _interp_to_grid(t, e, grid)
            if np.all(np.isnan(y)):
                continue
            curves.append(y)

        if len(curves) == 0:
            continue

        Y = np.vstack(curves)
        n = Y.shape[0]
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0, ddof=0)
        se = std / np.sqrt(n)
        ci_low = mean - ci_z * se
        ci_high = mean + ci_z * se

        out_rows.append(
            pd.DataFrame(
                {
                    "source": source_name,
                    "strategy": str(strategy),
                    "T": grid.astype(int),
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
        )

    if not out_rows:
        return pd.DataFrame(columns=["source", "strategy", "T", "n", "mean", "std", "ci_low", "ci_high"])

    return pd.concat(out_rows, ignore_index=True)


# -----------------------------
# Loading helpers (mostly unchanged)
# -----------------------------
def load_bafa_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        df["run_id"] = "NA"
    return df


def load_baselines_dir(baseline_dir: str, exclude_strategies=("random",)) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(baseline_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in baseline_dir={baseline_dir}")

    exclude = {s.lower() for s in exclude_strategies}

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["__file__"] = os.path.basename(f)

        # infer strategy if missing: trajectory_<strategy>_seedX.csv
        if "strategy" not in d.columns:
            base = os.path.basename(f)
            parts = base.replace(".csv", "").split("_")
            d["strategy"] = parts[1] if len(parts) >= 2 else "baseline"

        d["strategy"] = d["strategy"].astype(str)
        d = d[~d["strategy"].str.lower().isin(exclude)]
        if len(d):
            dfs.append(d)

    if not dfs:
        raise ValueError(f"After excluding {sorted(exclude)}, no baseline data remained in {baseline_dir}.")

    return pd.concat(dfs, ignore_index=True)


def load_cerm_ablation_dir(
    cerm_dir: str,
    *,
    force_strategy: str = "cerm",
) -> pd.DataFrame:
    """
    Loads all CSVs in cerm_dir and standardizes columns to match BAFA format.
    Expected columns include: run_id, seed, T_size, err_active (others ignored).
    If 'strategy' exists but is something else, we overwrite it.
    """
    files = sorted(glob.glob(os.path.join(cerm_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in cerm_dir={cerm_dir}")

    dfs = []
    for f in files:
        d = pd.read_csv(f)

        # Some exports include an unnamed index column; drop it if present
        if len(d.columns) and str(d.columns[0]).startswith("Unnamed"):
            d = d.drop(columns=[d.columns[0]])

        # Ensure run_id exists
        if "run_id" not in d.columns:
            d["run_id"] = os.path.basename(f).replace(".csv", "")

        # Ensure seed exists (fallback)
        if "seed" not in d.columns:
            d["seed"] = 0

        # Overwrite / set strategy to "cerm"
        d["strategy"] = force_strategy

        dfs.append(d)

    out = pd.concat(dfs, ignore_index=True)

    # sanity: require key cols
    _ensure_cols(out, ["run_id", "seed", "strategy", "T_size", "err_active"], "cerm_ablation")
    return out


# -----------------------------
# Plotting (unchanged)
# -----------------------------
def plot_mean_ci(agg: pd.DataFrame, outpath: str, title: str, y_label: str, x_label: str):
    """
    Line plot with mean + CI bands, ACL-ish sizing, no top/right frame,
    and a colorblind-friendly palette with NO green.
    """
    if agg.empty:
        raise ValueError("No aggregated data to plot.")

    label_map = {
        "ablation:cerm": "C-ERM (ablation)",
        "bafa:bo": "BAFA (BO)",
        "bafa:disagreement": "BAFA (disagreement)",
        "baseline:power": "Power sampling",
        "baseline:stratified": "Stratified sampling",
        "baseline:bo": "BO (ablation)",
    }

    color_map = {
        "bafa:bo": "#0072B2",
        "bafa:disagreement": "#ff7f0e",
        "ablation:bo": "#CC79A7",
        "baseline:power": "#56B4E9",
        "baseline:stratified": "#000000",
        "ablation:cerm": "#1ECCB5",
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)

    for (source, strategy), g in agg.groupby(["source", "strategy"]):
        g = g.sort_values("T")
        key = f"{source}:{strategy}"
        label = label_map.get(key, key)
        color = color_map.get(key, "#7f7f7f")

        if source == "bafa":
            linestyle, linewidth = "-", 2.0
        elif source == "ablation":
            linestyle, linewidth = "--", 1.5
        else:
            linestyle, linewidth = ":", 1.5

        x = g["T"].to_numpy()
        y = g["mean"].to_numpy()
        lo = g["ci_low"].to_numpy()
        hi = g["ci_high"].to_numpy()

        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)

    ax.set_xlabel(x_label, fontsize=9, fontfamily="serif")
    ax.set_ylabel(y_label, fontsize=9, fontfamily="serif")
    ax.set_ylim(0, 0.3)

    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.4)

    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.0,
        edgecolor="lightgray",
        fancybox=False,
    )

    fig.tight_layout(pad=0.3)

    pdf_path = outpath.replace(".png", ".pdf")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", format="pdf")
    fig.savefig(outpath, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)


# -----------------------------
# Metrics: "summarize_by_strategy"-style (ENHANCED)
# -----------------------------
def _normalized_auec(
    df: pd.DataFrame, t_col: str, err_col: str, t_max: int, prepend_t0: bool = True, t0_value: float = 0.0
) -> float:
    d = df[[t_col, err_col]].dropna().copy()
    if d.empty:
        return np.nan

    d[t_col] = pd.to_numeric(d[t_col], errors="coerce")
    d[err_col] = pd.to_numeric(d[err_col], errors="coerce")
    d = d.dropna(subset=[t_col, err_col])

    d = d.sort_values(t_col).drop_duplicates(subset=[t_col], keep="last")
    d = d[d[t_col] <= t_max]
    if d.empty:
        return np.nan

    if prepend_t0 and float(d[t_col].iloc[0]) > t0_value:
        first_err = float(d[err_col].iloc[0])
        d = pd.concat([pd.DataFrame({t_col: [t0_value], err_col: [first_err]}), d], ignore_index=True)

    if float(d[t_col].iloc[-1]) < t_max:
        last_err = float(d[err_col].iloc[-1])
        d = pd.concat([d, pd.DataFrame({t_col: [t_max], err_col: [last_err]})], ignore_index=True)

    t = d[t_col].to_numpy(dtype=float)
    e = d[err_col].to_numpy(dtype=float)
    return float(np.trapz(e, t) / float(t_max))


def _interp_err_at_budget(df: pd.DataFrame, t_col: str, err_col: str, budget: int) -> float:
    d = df[[t_col, err_col]].dropna().copy()

    if d.empty:
        return np.nan

    d[t_col] = pd.to_numeric(d[t_col], errors="coerce")
    d[err_col] = pd.to_numeric(d[err_col], errors="coerce")
    d = d.dropna(subset=[t_col, err_col]).sort_values(t_col)
    if d.empty:
        return np.nan

    if budget in d[t_col].values:
        return float(d.loc[d[t_col] == budget, err_col].iloc[-1])

    before = d[d[t_col] <= budget]
    after = d[d[t_col] >= budget]
    if len(before) == 0 or len(after) == 0:
        return np.nan

    t1, e1 = float(before[t_col].iloc[-1]), float(before[err_col].iloc[-1])
    t2, e2 = float(after[t_col].iloc[0]), float(after[err_col].iloc[0])

    if t1 == t2:
        return float(e1)
    return float(e1 + (e2 - e1) * (budget - t1) / (t2 - t1))


def to_long_timeseries(
    df: pd.DataFrame,
    *,
    source_name: str,
    t_col: str,
    err_col: str,
    strategy_col: str,
    seed_col: str,
    runid_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize raw trajectories to:
      T, err, seed, run_id, strategy
    where strategy is prefixed with source_name (e.g., 'bafa:bo', 'baseline:power').
    """
    d = df.copy()

    if seed_col not in d.columns:
        d[seed_col] = 0

    if runid_col is None or runid_col not in d.columns:
        d["run_id"] = "NA"
    else:
        d["run_id"] = d[runid_col].astype(str)

    d["strategy"] = source_name + ":" + d[strategy_col].astype(str)

    out = d.rename(columns={t_col: "T", err_col: "err", seed_col: "seed"})[
        ["T", "err", "seed", "run_id", "strategy"]
    ]
    return out


def summarize_by_strategy(
    ts: pd.DataFrame,
    *,
    eps_list=(0.02, 0.05),
    t_max=1000,
    budget=250,
    min_T_for_metrics=36,
    t_col="T",
    err_col="err",
    seed_col="seed",
    strategy_col="strategy",
) -> pd.DataFrame:
    """
    Enhanced summarize_by_strategy with full descriptive statistics including:
    - Mean, std, median, IQR, min, max, n
    - 95% confidence intervals for all metrics
    """
    df = ts.copy()

    for c in [t_col, err_col, seed_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[t_col, err_col, seed_col, strategy_col])

    if "run_id" not in df.columns:
        df["run_id"] = "NA"

    df = df.sort_values([strategy_col, seed_col, "run_id", t_col])
    df = df.drop_duplicates(subset=[strategy_col, seed_col, "run_id", t_col], keep="last")

    out_rows = []

    for strategy, sdf_all in df.groupby(strategy_col, dropna=False):
        replicates = sdf_all[[seed_col, "run_id"]].drop_duplicates().values.tolist()

        auec_vals = []
        errB_vals = []
        first_hit = {eps: [] for eps in eps_list}

        for seed, run_id in replicates:
            rdf = sdf_all[(sdf_all[seed_col] == seed) & (sdf_all["run_id"] == run_id)].sort_values(t_col)

            auec_vals.append(_normalized_auec(rdf, t_col=t_col, err_col=err_col, t_max=t_max))
            errB_vals.append(_interp_err_at_budget(rdf, t_col=t_col, err_col=err_col, budget=budget))

            rdf_validT = rdf[rdf[t_col] >= float(min_T_for_metrics)]
            for eps in eps_list:
                hit = rdf_validT[rdf_validT[err_col] <= eps]
                first_hit[eps].append(float(hit[t_col].iloc[0]) if len(hit) else np.nan)

        pop_curve = (
            sdf_all.groupby(t_col)[err_col].mean().reset_index().sort_values(t_col)
        )
        pop_curve = pop_curve[pop_curve[t_col] >= float(min_T_for_metrics)]

        # Compute descriptive statistics for AUEC
        auec_stats = _descriptive_stats(np.array(auec_vals))
        
        # Compute descriptive statistics for error at budget
        errB_stats = _descriptive_stats(np.array(errB_vals))

        base = {
            "strategy": strategy,
            "n_replicates": int(len(replicates)),
            # AUEC metrics with full stats
            "auec_mean": auec_stats["mean"],
            "auec_std": auec_stats["std"],
            "auec_median": auec_stats["median"],
            "auec_iqr": auec_stats["iqr"],
            "auec_min": auec_stats["min"],
            "auec_max": auec_stats["max"],
            "auec_ci_low": auec_stats["ci_low"],
            "auec_ci_high": auec_stats["ci_high"],
            # Error at budget metrics with full stats
            f"error_at_{budget}_mean": errB_stats["mean"],
            f"error_at_{budget}_std": errB_stats["std"],
            f"error_at_{budget}_median": errB_stats["median"],
            f"error_at_{budget}_iqr": errB_stats["iqr"],
            f"error_at_{budget}_min": errB_stats["min"],
            f"error_at_{budget}_max": errB_stats["max"],
            f"error_at_{budget}_ci_low": errB_stats["ci_low"],
            f"error_at_{budget}_ci_high": errB_stats["ci_high"],
        }

        for eps in eps_list:
            vals = np.array(first_hit[eps], dtype=float)
            hit_stats = _descriptive_stats(vals)

            crossed = pop_curve[pop_curve[err_col] <= eps]
            mean_crossing_T = float(crossed[t_col].min()) if len(crossed) else np.nan

            out_rows.append(
                {
                    **base,
                    "epsilon": float(eps),
                    "mean_crossing_T": mean_crossing_T,
                    # Full statistics for queries to epsilon
                    f"queries_to_{eps}_mean": hit_stats["mean"],
                    f"queries_to_{eps}_std": hit_stats["std"],
                    f"queries_to_{eps}_median": hit_stats["median"],
                    f"queries_to_{eps}_iqr": hit_stats["iqr"],
                    f"queries_to_{eps}_min": hit_stats["min"],
                    f"queries_to_{eps}_max": hit_stats["max"],
                    f"queries_to_{eps}_ci_low": hit_stats["ci_low"],
                    f"queries_to_{eps}_ci_high": hit_stats["ci_high"],
                    f"queries_to_{eps}_n_reached": hit_stats["n"],
                }
            )

    return pd.DataFrame(out_rows).sort_values(["strategy", "epsilon"]).reset_index(drop=True)


def generate_appendix_table(
    ts: pd.DataFrame,
    *,
    budgets: List[int] = [100, 250, 500, 1000],
    t_col: str = "T",
    err_col: str = "err",
    seed_col: str = "seed",
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    """
    Generate a comprehensive appendix table with error statistics at multiple budgets.
    Returns a wide-format table suitable for LaTeX inclusion.
    """
    df = ts.copy()
    
    for c in [t_col, err_col, seed_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[t_col, err_col, seed_col, strategy_col])
    
    if "run_id" not in df.columns:
        df["run_id"] = "NA"
    
    df = df.sort_values([strategy_col, seed_col, "run_id", t_col])
    df = df.drop_duplicates(subset=[strategy_col, seed_col, "run_id", t_col], keep="last")
    
    out_rows = []
    
    for strategy, sdf_all in df.groupby(strategy_col, dropna=False):
        replicates = sdf_all[[seed_col, "run_id"]].drop_duplicates().values.tolist()
        
        row = {"strategy": strategy, "n_replicates": len(replicates)}
        
        for budget in budgets:
            budget_vals = []
            for seed, run_id in replicates:
                rdf = sdf_all[(sdf_all[seed_col] == seed) & (sdf_all["run_id"] == run_id)].sort_values(t_col)
                budget_vals.append(_interp_err_at_budget(rdf, t_col=t_col, err_col=err_col, budget=budget))
            
            stats = _descriptive_stats(np.array(budget_vals))
            row[f"T{budget}_mean"] = stats["mean"]
            row[f"T{budget}_ci_low"] = stats["ci_low"]
            row[f"T{budget}_ci_high"] = stats["ci_high"]
            row[f"T{budget}_median"] = stats["median"]
            row[f"T{budget}_iqr"] = stats["iqr"]
        
        out_rows.append(row)
    
    return pd.DataFrame(out_rows)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Aggregate BAFA + baselines (+ optional ablations) into mean/std/95% CI curves and plot together."
    )

    ap.add_argument("--bafa_csv", type=str, default="../BAFA/bafa_trajectories.csv")
    ap.add_argument("--baseline_dir", type=str, default="../baselines/results_jigsaw")
    ap.add_argument("--cerm_dir", type=str, default="../ablations")
    ap.add_argument("--include_cerm", action="store_true", help="Include C-ERM ablation trajectories.")
    ap.add_argument("--outdir", type=str, default="agg_jigsaw")
    ap.add_argument("--t_max", type=int, default=1000)
    ap.add_argument("--t_start", type=int, default=20)
    ap.add_argument("--grid_step", type=int, default=1)

    ap.add_argument("--bafa_t_col", type=str, default="T_size")
    ap.add_argument("--bafa_err_col", type=str, default="err_active")
    ap.add_argument("--bafa_strategy_col", type=str, default="strategy")
    ap.add_argument("--bafa_seed_col", type=str, default="seed")
    ap.add_argument("--bafa_runid_col", type=str, default="run_id")

    ap.add_argument("--base_t_col", type=str, default="t_size")
    ap.add_argument("--base_err_col", type=str, default="abs_error_roc_auc_diff")
    ap.add_argument("--base_strategy_col", type=str, default="strategy")
    ap.add_argument("--base_seed_col", type=str, default="seed")

    ap.add_argument(
        "--eps",
        type=str,
        default="0.02,0.05",
        help="Comma-separated epsilon thresholds (per-replicate first-hit + mean crossing).",
    )
    ap.add_argument("--target_budget", type=int, default=250, help="Budget for error_at_B summary (per-replicate).")
    ap.add_argument(
        "--min_T_for_metrics",
        type=int,
        default=36,
        help="Minimum query budget T before counting crossings/hits in metrics (e.g., ignore T<36).",
    )
    
    ap.add_argument(
        "--appendix_budgets",
        type=str,
        default="100,250,500,1000",
        help="Comma-separated budgets for appendix table.",
    )

    # NEW: exclude strategies everywhere (raw names or prefixed like 'baseline:power')
    ap.add_argument(
        "--exclude_strategies",
        type=str,
        default="random",
        help="Comma-separated strategies to exclude. Can be raw (e.g., 'random') or prefixed (e.g., 'baseline:power').",
    )

    ap.add_argument("--title", type=str, default="Mean abs error ± 95% CI (first 1000 queries)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eps_list = tuple(float(x.strip()) for x in args.eps.split(",") if x.strip())
    exclude_list = _parse_exclude_list(args.exclude_strategies)
    appendix_budgets = [int(x.strip()) for x in args.appendix_budgets.split(",") if x.strip()]

    # ---- Load
    bafa = load_bafa_csv(args.bafa_csv)
    baselines = load_baselines_dir(args.baseline_dir, exclude_strategies=tuple(exclude_list))

    # Apply excludes to raw BAFA too (matches raw or "bafa:<raw>")
    bafa = _apply_strategy_excludes(
        bafa,
        strategy_col=args.bafa_strategy_col,
        exclude=exclude_list,
        source_prefix="bafa",
    )

    # Baselines already filtered in loader by raw name; also allow "baseline:<raw>" filters
    baselines = _apply_strategy_excludes(
        baselines,
        strategy_col=args.base_strategy_col,
        exclude=exclude_list,
        source_prefix="baseline",
    )

    # ---- Aggregate BAFA (for plotting)
    agg_bafa = aggregate_curves(
        bafa,
        source_name="bafa",
        t_col=args.bafa_t_col,
        err_col=args.bafa_err_col,
        strategy_col=args.bafa_strategy_col,
        replicate_cols=[args.bafa_seed_col, args.bafa_runid_col],
        t_start=args.t_start,
        t_max=args.t_max,
        grid_step=args.grid_step,
    )

    # ---- Aggregate baselines (for plotting)
    agg_base = aggregate_curves(
        baselines,
        source_name="baseline",
        t_col=args.base_t_col,
        err_col=args.base_err_col,
        strategy_col=args.base_strategy_col,
        replicate_cols=[args.base_seed_col],
        t_start=args.t_start,
        t_max=args.t_max,
        grid_step=args.grid_step,
    )

    aggs = [agg_bafa, agg_base]

    # ---- Optional: C-ERM ablation
    cerm = None
    if args.include_cerm:
        cerm = load_cerm_ablation_dir(args.cerm_dir, force_strategy="cerm")
        # Allow excluding "cerm" via raw or "ablation:cerm"
        cerm = _apply_strategy_excludes(
            cerm,
            strategy_col="strategy",
            exclude=exclude_list,
            source_prefix="ablation",
        )

        agg_cerm = aggregate_curves(
            cerm,
            source_name="ablation",
            t_col="T_size",
            err_col="err_active",
            strategy_col="strategy",
            replicate_cols=["seed", "run_id"],
            t_start=args.t_start,
            t_max=args.t_max,
            grid_step=args.grid_step,
        )
        aggs.append(agg_cerm)

    agg_combined = pd.concat(aggs, ignore_index=True)

    # ---- Save aggregated curves
    agg_bafa.to_csv(outdir / "agg_bafa.csv", index=False)
    agg_base.to_csv(outdir / "agg_baselines.csv", index=False)
    if args.include_cerm:
        agg_cerm.to_csv(outdir / "agg_cerm.csv", index=False)
    agg_combined.to_csv(outdir / "agg_combined.csv", index=False)

    # ---- Compute metrics from RAW trajectories (summarize_by_strategy-style)
    ts_parts = []

    ts_parts.append(
        to_long_timeseries(
            bafa,
            source_name="bafa",
            t_col=args.bafa_t_col,
            err_col=args.bafa_err_col,
            strategy_col=args.bafa_strategy_col,
            seed_col=args.bafa_seed_col,
            runid_col=args.bafa_runid_col,
        )
    )

    ts_parts.append(
        to_long_timeseries(
            baselines,
            source_name="baseline",
            t_col=args.base_t_col,
            err_col=args.base_err_col,
            strategy_col=args.base_strategy_col,
            seed_col=args.base_seed_col,
            runid_col=None,
        )
    )

    if args.include_cerm and cerm is not None:
        ts_parts.append(
            to_long_timeseries(
                cerm,
                source_name="ablation",
                t_col="T_size",
                err_col="err_active",
                strategy_col="strategy",
                seed_col="seed",
                runid_col="run_id",
            )
        )

    ts_all = pd.concat(ts_parts, ignore_index=True)

    # Filter again at the prefixed level (now strategies look like "bafa:bo", "baseline:power", ...)
    ts_all = _apply_strategy_excludes(
        ts_all,
        strategy_col="strategy",
        exclude=exclude_list,
        source_prefix=None,  # already prefixed in the column
    )

    # Enhanced metrics with full descriptive statistics
    metrics_df = summarize_by_strategy(
        ts_all,
        eps_list=eps_list,
        t_max=args.t_max,
        budget=args.target_budget,
        min_T_for_metrics=args.min_T_for_metrics,
        t_col="T",
        err_col="err",
        seed_col="seed",
        strategy_col="strategy",
    )

    metrics_file = outdir / "convergence_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    # Generate appendix table with multiple budgets
    appendix_df = generate_appendix_table(
        ts_all,
        budgets=appendix_budgets,
        t_col="T",
        err_col="err",
        seed_col="seed",
        strategy_col="strategy",
    )
    
    appendix_file = outdir / "appendix_table.csv"
    appendix_df.to_csv(appendix_file, index=False)

    print(f"\nSaved convergence metrics (with CI): {metrics_file}")
    print(f"Saved appendix table: {appendix_file}")
    print("\nConvergence Metrics (sample):")
    print(metrics_df[["strategy", "epsilon", "n_replicates", "auec_mean", "auec_ci_low", "auec_ci_high"]].to_string(index=False))
    print("\nAppendix Table (sample):")
    print(appendix_df.head().to_string(index=False))

    # ---- Plot
    plot_mean_ci(
        agg_combined,
        outpath=str(outdir / "mean_ci_plot.png"),
        title=args.title,
        y_label="abs error (ΔAUC)",
        x_label="queries (T)",
    )

    print("\n[OK] Wrote:")
    print(f"- {outdir/'agg_bafa.csv'}")
    print(f"- {outdir/'agg_baselines.csv'}")
    print(f"- {outdir/'agg_combined.csv'}")
    print(f"- {outdir/'convergence_metrics.csv'}")
    print(f"- {outdir/'appendix_table.csv'}")
    print(f"- {outdir/'mean_ci_plot.png'}")
    if args.include_cerm:
        print(f"- {outdir/'agg_cerm.csv'}")


if __name__ == "__main__":
    main()