#!/usr/bin/env python3
"""
Calculate uncertainty coverage for BAFA trajectories.

Coverage = What percentage of cases have the true value within [h_min, h_max]?

Usage:
    python calculate_coverage.py --csv your_data.csv --out output_name

Output:
    - Prints coverage statistics to console
    - Saves plot as output_name.png and output_name.pdf
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.lines import Line2D


def calculate_coverage(df, strategy_name):
    """
    Calculate coverage for one strategy.
    
    Coverage = fraction of cases where: delta_bb_init ∈ [h_min, h_max]
    """
    # Check if true value falls within predicted bounds
    in_bounds = (df["delta_bb_init"] >= df["h_min"]) & (df["delta_bb_init"] <= df["h_max"])
    coverage = in_bounds.mean()
    
    # Count how many cases are in/out of bounds
    n_in = in_bounds.sum()
    n_out = (~in_bounds).sum()
    n_total = len(df)
    
    return {
        'coverage': coverage,
        'n_in_bounds': n_in,
        'n_out_bounds': n_out,
        'n_total': n_total,
        'strategy': strategy_name
    }


def compute_all_stats(df):
    """Compute coverage and correlation statistics for each strategy."""
    results = {}
    
    for strategy, group in df.groupby("strategy"):
        # Correlation between bound width and absolute error
        pearson = pearsonr(group["width_abs"], group["err_active"])
        spearman = spearmanr(group["width_abs"], group["err_active"])
        
        # Coverage calculation
        cov = calculate_coverage(group, strategy)
        
        # Error-to-width ratio
        ratio = (2 * group["err_active"] / group["width_abs"]).replace([np.inf, -np.inf], np.nan).dropna()
        
        results[strategy] = {
            'n': len(group),
            'coverage': cov['coverage'],
            'n_in_bounds': cov['n_in_bounds'],
            'n_out_bounds': cov['n_out_bounds'],
            'pearson_r': pearson.statistic,
            'pearson_p': pearson.pvalue,
            'spearman_rho': spearman.correlation,
            'spearman_p': spearman.pvalue,
            'ratio_mean': float(ratio.mean()),
            'ratio_median': float(ratio.median()),
            'ratio_q25': float(ratio.quantile(0.25)),
            'ratio_q75': float(ratio.quantile(0.75)),
        }
    
    return results


def print_statistics(stats):
    """Print coverage and correlation statistics."""
    print("\n" + "=" * 70)
    print("UNCERTAINTY CALIBRATION DIAGNOSTICS")
    print("=" * 70)
    
    for strategy, s in stats.items():
        print(f"\n{strategy.upper()} Strategy (n={s['n']}):")
        print("-" * 70)
        
        # Coverage
        print(f"Coverage (true value ∈ [h_min, h_max]):")
        print(f"  • {s['coverage']*100:.1f}% ({s['n_in_bounds']}/{s['n']} cases)")
        print(f"  • {s['n_out_bounds']} cases where true value falls OUTSIDE predicted bounds")
        
        # Correlations
        print(f"\nCorrelation (bound width vs absolute error):")
        print(f"  • Pearson r = {s['pearson_r']:.3f} (p={s['pearson_p']:.2e})")
        print(f"  • Spearman ρ = {s['spearman_rho']:.3f} (p={s['spearman_p']:.2e})")
        
        # Error-to-width ratio
        print(f"\nError-to-Width Ratio (2×error/width):")
        print(f"  • Mean: {s['ratio_mean']:.3f}")
        print(f"  • Median: {s['ratio_median']:.3f}")
        print(f"  • Q25-Q75: [{s['ratio_q25']:.3f}, {s['ratio_q75']:.3f}]")
        
        # Interpretation
        print(f"\nInterpretation:")
        if s['coverage'] >= 0.95:
            print(f"  ✓ Excellent calibration! Bounds capture true value in {s['coverage']*100:.1f}% of cases.")
        elif s['coverage'] >= 0.80:
            print(f"  ~ Good calibration. Bounds capture true value in {s['coverage']*100:.1f}% of cases.")
        elif s['coverage'] >= 0.60:
            print(f"  ⚠ Moderate calibration. Bounds miss true value in {100-s['coverage']*100:.1f}% of cases.")
        else:
            print(f"  ✗ Poor calibration! Bounds miss true value in {100-s['coverage']*100:.1f}% of cases.")
            print(f"    → Predicted uncertainty intervals are too narrow/overconfident.")


def make_plot(df, output_prefix):
    """Create scatter plot with regression lines."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    
    colors = {
        "bo": "#0173B2",
        "disagreement": "#DE8F05",
    }
    
    strategies = [s for s in ["bo", "disagreement"] if s in df["strategy"].unique()]
    if not strategies:
        strategies = sorted(df["strategy"].unique())
    
    correlations = {}
    
    for strategy in strategies:
        group = df[df["strategy"] == strategy]
        
        # Calculate correlation
        corr, p_value = pearsonr(group["width_abs"], group["err_active"])
        correlations[strategy] = (corr, p_value, len(group))
        
        # Scatter plot
        ax.scatter(
            group["width_abs"], 
            group["err_active"],
            alpha=0.35, 
            s=18,
            color=colors.get(strategy, "C0"),
            edgecolors="none",
            rasterized=True,
        )
        
        # Regression line
        z = np.polyfit(group["width_abs"], group["err_active"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, group["width_abs"].max(), 200)
        ax.plot(
            x_line, 
            p(x_line),
            color=colors.get(strategy, "C0"),
            linewidth=2.0,
            alpha=0.9,
        )
    
    # Formatting
    ax.set_xlabel(r"Bound Width ($\mu_{\max}-\mu_{\min}$)", fontsize=10)
    ax.set_ylabel(r"Absolute Error ($|\widehat{\Delta \mathrm{AUC}}-\Delta \mathrm{AUC}|$)", fontsize=10)
    
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.003)
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", length=3)
    
    # Legend
    legend_elements = []
    for strategy in strategies:
        corr, _, n = correlations[strategy]
        label = f'BAFA ({"BO" if strategy=="bo" else "disagreement"}): $r={corr:.2f}$'
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color=colors.get(strategy, "C0"),
                markerfacecolor=colors.get(strategy, "C0"),
                markersize=5.5,
                label=label,
                linestyle="-",
                linewidth=2.0,
                markeredgewidth=0,
            )
        )
    
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8.5,
        frameon=False,
        handlelength=2.0,
    )
    
    plt.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", dpi=600, bbox_inches="tight")
    print(f"\n✓ Saved plots: {output_prefix}.png and {output_prefix}.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate coverage and create plots for BAFA trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # If delta_bb_init is in the CSV:
    python calculate_coverage.py --csv bafa_trajectories_bios.csv --out bios_analysis
    
    # If delta_bb_init needs to be merged from eval file:
    python calculate_coverage.py --csv bafa_trajectories_jigsaw.csv --eval eval_file.csv --out jigsaw_analysis
    
Required CSV columns:
    - strategy: "bo" or "disagreement"
    - h_min: Lower bound of uncertainty interval
    - h_max: Upper bound of uncertainty interval
    - delta_bb_init: True value (ground truth)
    - width_abs: Width of uncertainty interval (h_max - h_min)
    - err_active: Absolute error |estimate - true_value|
        """
    )
    
    parser.add_argument("--csv", required=True, help="Path to CSV file with trajectory data")
    parser.add_argument("--eval", help="Optional: Path to eval CSV with delta_bb_init (if not in main CSV)")
    parser.add_argument("--out", default="coverage_analysis", help="Output file prefix (no extension)")
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    
    # If delta_bb_init is missing and eval file is provided, merge it
    if 'delta_bb_init' not in df.columns and args.eval:
        print(f"\nMerging with eval file: {args.eval}")
        eval_df = pd.read_csv(args.eval)
        merge_cols = ['run_id', 'seed', 'strategy', '_step', 'iter']
        df = df.merge(
            eval_df[merge_cols + ['delta_bb_init']], 
            on=merge_cols, 
            how='left',
            suffixes=('', '_eval')
        )
        print(f"After merge: {len(df)} rows, {df['delta_bb_init'].notna().sum()} with delta_bb_init")
    
    # Check required columns
    required = {"strategy", "h_min", "h_max", "delta_bb_init", "width_abs", "err_active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {df.columns.tolist()}")
    
    # Compute statistics
    stats = compute_all_stats(df)
    
    # Print results
    print_statistics(stats)
    
    # Create plot
    make_plot(df, args.out)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()