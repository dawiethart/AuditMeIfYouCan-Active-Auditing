#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.lines import Line2D

def compute_stats(df: pd.DataFrame):
    out = {}
    for strategy, g in df.groupby("strategy"):
        pr = pearsonr(g["width_abs"], g["err_active"])
        sr = spearmanr(g["width_abs"], g["err_active"])
        coverage = (g["err_active"] <= g["width_abs"] / 2).mean()

        ratio = (2 * g["err_active"] / g["width_abs"]).replace([np.inf, -np.inf], np.nan).dropna()

        out[strategy] = {
            "n": len(g),
            "pearson_r": pr.statistic,
            "pearson_p": pr.pvalue,
            "spearman_rho": sr.correlation,
            "spearman_p": sr.pvalue,
            "coverage": coverage,
            "ratio_mean": float(ratio.mean()),
            "ratio_median": float(ratio.median()),
            "ratio_q75": float(ratio.quantile(0.75)),
        }
    return out

def make_plot(df: pd.DataFrame, out_prefix: str):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))  # ACL-ish single-column feel

    colors = {
        "bo": "#0173B2",
        "disagreement": "#DE8F05",
    }

    # Keep a stable order
    strategies = [s for s in ["bo", "disagreement"] if s in df["strategy"].unique()]
    if not strategies:
        strategies = sorted(df["strategy"].unique())

    correlations = {}
    for strategy in strategies:
        g = df[df["strategy"] == strategy]

        corr, p_value = pearsonr(g["width_abs"], g["err_active"])
        correlations[strategy] = (corr, p_value, len(g))

        ax.scatter(
            g["width_abs"], g["err_active"],
            alpha=0.35, s=18,
            color=colors.get(strategy, "C0"),
            edgecolors="none",
            rasterized=True,
        )

        # Regression line
        z = np.polyfit(g["width_abs"], g["err_active"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, g["width_abs"].max(), 200)
        ax.plot(
            x_line, p(x_line),
            color=colors.get(strategy, "C0"),
            linewidth=2.0,
            alpha=0.9,
        )

    ax.set_xlabel(r"Bound Width ($\mu_{\max}-\mu_{\min}$)", fontsize=10)
    ax.set_ylabel(r"Absolute Error ($|\widehat{\Delta \mathrm{AUC}}-\Delta \mathrm{AUC}|$)", fontsize=10)

    # ACL-ish cosmetics
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.003)

    # Remove spines (user requested left + top)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # (Optional) keep ticks readable even without left spine
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", length=3)

    # Legend (no frame)
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
        frameon=False,   # remove legend frame
        handlelength=2.0,
    )

    plt.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", dpi=600,bbox_inches="tight")
    print(f"✓ Saved: {out_prefix}.png and {out_prefix}.pdf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to bafa_trajectories.csv")
    ap.add_argument("--out", default="figure5_single_panel", help="Output prefix (no extension)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = {"strategy", "width_abs", "err_active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stats = compute_stats(df)
    print("\n=== Uncertainty diagnostics (appendix) ===")
    for strat, s in stats.items():
        print(f"\n{strat.upper()} (n={s['n']}):")
        print(f"  Pearson r = {s['pearson_r']:.3f}")
        print(f"  Spearman rho = {s['spearman_rho']:.3f}")
        print(f"  Coverage (|e| <= w/2) = {100*s['coverage']:.1f}%")
        print(f"  Ratio 2|e|/w: mean={s['ratio_mean']:.3f}, median={s['ratio_median']:.3f}, p75={s['ratio_q75']:.3f}")

    make_plot(df, args.out)

if __name__ == "__main__":
    main()
