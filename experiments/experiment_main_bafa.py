#!/usr/bin/env python3
"""
Main BAFA experiments: Run all strategies (BO, Disagreement, C-ERM ablation) 
across 20 seeds for both case studies.

Usage:
  # CivilComments (Case Study A)
  python experiment_main_bafa.py --dataset jigsaw --blackbox hatebert
  
  # Bias-in-Bios (Case Study B)
  python experiment_main_bafa.py --dataset bios --blackbox bios_csv \
    --bios_scores_csv blackbox_bios.csv --bios_max_rows 50000
"""
import argparse
import subprocess
import sys
from itertools import product

# Default seeds for reproducibility (20 seeds as per paper)
DEFAULT_SEEDS = list(range(20))

# Strategies to evaluate
STRATEGIES = {
    "bo": "BAFA with Bayesian Optimization",
    "disagreement": "BAFA with Disagreement sampling", 
    "stratified": "Stratified sampling baseline (C-ERM only)"
}


def build_cmd(
    python_exe: str,
    dataset: str,
    blackbox: str,
    strategy: str,
    seed: int,
    title_prefix: str,
    bios_scores_csv: str | None = None,
    bios_targets: list[str] | None = None,
    bios_max_rows: int | None = None,
):
    """Build command for a single run with paper-specified hyperparameters."""
    
    # Dataset-specific defaults from Table 7
    if dataset.lower() == "jigsaw":
        epochs_opt = 10
        batch_size = 512
        model = "bert-base-uncased"
    else:  # bios
        epochs_opt = 8
        batch_size = 512
        model = "bert-base-uncased"
    
    title = f"{title_prefix}_{strategy}_seed{seed}"
    
    cmd = [
        python_exe, "main.py",
        "--dataset", dataset,
        "--blackbox", blackbox,
        "--strategy", strategy,
        "--model", model,
        
        # Query and iteration settings (Table 7)
        "--size_T", "16",           # Top-k batch size (queries per round)
        "--iterations", "75",        # Total audit rounds
        
        # Surrogate training (Table 7)
        "--epochs_sur", "4",         # Per-round surrogate fine-tuning
        "--epochs_opt", str(epochs_opt),  # C-ERM gradient steps
        "--batch_size", str(batch_size),  # C-ERM batch size
        
        # C-ERM constraints (Table 7)
        "--lambda_penalty", "1e-2",  # Constraint tolerance λ
        "--epsilon", "1e-2",         # Target precision (stopping)
        "--reg_alpha", "2.0",        # Distribution matching penalty
        
        # BO-specific parameters (Table 7, only used if strategy=bo)
        "--bo_beta", "1.0",          # UCB exploration parameter
        "--bo_diversity_gamma", "0.2",  # Diversity penalty weight
        "--bo_acq", "ucb",           # Acquisition function
        
        "--k_batch", "16",           # Same as size_T for consistency
        "--seed", str(seed),
        "--title", title,
    ]
    
    # Bias-in-Bios specific arguments
    if dataset.lower() == "bios":
        if not bios_scores_csv:
            raise ValueError("--bios_scores_csv required for bios dataset")
        cmd += ["--bios_scores_csv", bios_scores_csv]
        
        if bios_max_rows:
            cmd += ["--bios_max_rows", str(bios_max_rows)]
        
        if bios_targets:
            cmd += ["--bios_targets"] + list(bios_targets)
    
    return cmd


def main():
    p = argparse.ArgumentParser(
        description="Run main BAFA experiments across all strategies and seeds"
    )
    
    # Dataset selection
    p.add_argument(
        "--dataset", 
        type=str, 
        default="jigsaw",
        choices=["jigsaw", "bios"],
        help="Case study dataset"
    )
    p.add_argument(
        "--blackbox",
        type=str,
        default="hatebert",
        choices=["hatebert", "bert", "perspective_offline", "bios_csv"],
        help="Black-box model to audit"
    )
    
    # Strategy selection
    p.add_argument(
        "--strategies",
        nargs="*",
        choices=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        help="Which strategies to run (default: all)"
    )
    
    # Seed configuration
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Custom seed list (default: 0-19)"
    )
    p.add_argument(
        "--num_seeds",
        type=int,
        default=20,
        help="Number of seeds if using auto-generated range"
    )
    
    # Bias-in-Bios specific
    p.add_argument(
        "--bios_scores_csv",
        type=str,
        help="Path to cached black-box scores for Bias-in-Bios"
    )
    p.add_argument(
        "--bios_max_rows",
        type=int,
        default=50000,
        help="Subsample size for Bias-in-Bios (default: 50k as per paper)"
    )
    p.add_argument(
        "--bios_targets",
        nargs="*",
        default=["professor"],
        help="Target occupations for one-vs-rest (default: professor)"
    )
    
    # Execution control
    p.add_argument(
        "--title_prefix",
        type=str,
        default="main_bafa",
        help="Prefix for run titles in W&B"
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    
    args = p.parse_args()
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(args.num_seeds))
    
    python_exe = sys.executable
    
    # Build all commands
    cmds = []
    for strategy, seed in product(args.strategies, seeds):
        cmd = build_cmd(
            python_exe=python_exe,
            dataset=args.dataset,
            blackbox=args.blackbox,
            strategy=strategy,
            seed=seed,
            title_prefix=args.title_prefix,
            bios_scores_csv=args.bios_scores_csv,
            bios_targets=args.bios_targets,
            bios_max_rows=args.bios_max_rows,
        )
        cmds.append(cmd)
    
    # Summary
    print(f"=== BAFA Main Experiments ===")
    print(f"Dataset: {args.dataset}")
    print(f"Strategies: {args.strategies}")
    print(f"Seeds: {len(seeds)} ({min(seeds)}-{max(seeds)})")
    print(f"Total runs: {len(cmds)}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Commands to execute:")
        for i, c in enumerate(cmds, 1):
            print(f"\n[{i}/{len(cmds)}]")
            print(" ".join(map(str, c)))
        return
    
    # Execute
    for i, c in enumerate(cmds, 1):
        print(f"\n{'='*60}")
        print(f"Running [{i}/{len(cmds)}]")
        print(" ".join(map(str, c)))
        print(f"{'='*60}\n")
        
        try:
            subprocess.run(c, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n!!! Run failed with exit code {e.returncode} !!!")
            print("Continue? (y/n): ", end="")
            if input().lower() != 'y':
                print("Stopping execution.")
                sys.exit(1)


if __name__ == "__main__":
    main()
