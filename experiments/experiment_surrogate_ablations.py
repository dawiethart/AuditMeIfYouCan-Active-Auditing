#!/usr/bin/env python3
"""
Surrogate Model Ablations: Test BAFA with different surrogate architectures.

From paper Appendix A.6.1: "replacing the surrogate with DistilBERT increases 
AUEC by less than 15%"

This script evaluates:
- bert-base-uncased (baseline, 110M params)
- distilbert-base-uncased (smaller, 66M params)  
- roberta-large (larger, 355M params)

Usage:
  # Run all surrogate ablations for CivilComments
  python experiment_surrogate_ablations.py --dataset jigsaw --blackbox hatebert
  
  # Test specific surrogate
  python experiment_surrogate_ablations.py --surrogates distilbert-base-uncased
"""
import argparse
import subprocess
import sys
from itertools import product

# Surrogate models to test
SURROGATES = {
    "bert-base-uncased": "Baseline (110M params)",
    "distilbert-base-uncased": "Smaller variant (66M params)",
    "roberta-large": "Larger variant (355M params)",
}

# Use fewer seeds for ablations (computational cost)
DEFAULT_SEEDS = list(range(5))


def build_cmd(
    python_exe: str,
    dataset: str,
    blackbox: str,
    surrogate_model: str,
    strategy: str,
    seed: int,
    title_prefix: str,
    bios_scores_csv: str | None = None,
    bios_max_rows: int | None = None,
):
    """Build command with specified surrogate model."""
    
    # Dataset-specific defaults
    if dataset.lower() == "jigsaw":
        epochs_opt = 10
        batch_size = 512
    else:  # bios
        epochs_opt = 8
        batch_size = 512
    
    # Adjust batch size for larger models to fit in memory
    if "large" in surrogate_model.lower():
        batch_size = min(batch_size, 256)
        print(f"  [Note: Reduced batch_size to {batch_size} for {surrogate_model}]")
    
    title = f"{title_prefix}_{strategy}_{surrogate_model.replace('/', '_')}_seed{seed}"
    
    cmd = [
        python_exe, "main.py",
        "--dataset", dataset,
        "--blackbox", blackbox,
        "--strategy", strategy,
        "--model", surrogate_model,  # CHANGED: surrogate model
        
        # Standard BAFA settings (Table 7)
        "--size_T", "16",
        "--iterations", "75",
        "--epochs_sur", "4",
        "--epochs_opt", str(epochs_opt),
        "--batch_size", str(batch_size),
        "--lambda_penalty", "1e-2",
        "--epsilon", "1e-2",
        "--reg_alpha", "2.0",
        "--k_batch", "16",
        
        # BO params (if strategy=bo)
        "--bo_beta", "1.0",
        "--bo_diversity_gamma", "0.2",
        "--bo_acq", "ucb",
        
        "--seed", str(seed),
        "--title", title,
    ]
    
    # Bias-in-Bios specific
    if dataset.lower() == "bios":
        if not bios_scores_csv:
            raise ValueError("--bios_scores_csv required for bios dataset")
        cmd += ["--bios_scores_csv", bios_scores_csv]
        if bios_max_rows:
            cmd += ["--bios_max_rows", str(bios_max_rows)]
    
    return cmd


def main():
    p = argparse.ArgumentParser(
        description="Surrogate model ablation experiments"
    )
    
    p.add_argument(
        "--dataset",
        type=str,
        default="jigsaw",
        choices=["jigsaw", "bios"]
    )
    p.add_argument(
        "--blackbox",
        type=str,
        default="hatebert",
        choices=["hatebert", "bert", "perspective_offline", "bios_csv"]
    )
    p.add_argument(
        "--surrogates",
        nargs="*",
        choices=list(SURROGATES.keys()),
        default=list(SURROGATES.keys()),
        help="Which surrogate models to test"
    )
    p.add_argument(
        "--strategies",
        nargs="*",
        choices=["bo", "disagreement"],
        default=["disagreement"],  # Test disagreement first (faster)
        help="Which BAFA strategies to run"
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Custom seeds (default: 0-4 for efficiency)"
    )
    p.add_argument(
        "--bios_scores_csv",
        type=str,
        help="Cached scores for Bias-in-Bios"
    )
    p.add_argument(
        "--bios_max_rows",
        type=int,
        default=50000
    )
    p.add_argument(
        "--title_prefix",
        type=str,
        default="surrogate_ablation"
    )
    p.add_argument(
        "--dry_run",
        action="store_true"
    )
    
    args = p.parse_args()
    
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    python_exe = sys.executable
    
    # Build commands
    cmds = []
    for surrogate, strategy, seed in product(
        args.surrogates, args.strategies, seeds
    ):
        cmd = build_cmd(
            python_exe=python_exe,
            dataset=args.dataset,
            blackbox=args.blackbox,
            surrogate_model=surrogate,
            strategy=strategy,
            seed=seed,
            title_prefix=args.title_prefix,
            bios_scores_csv=args.bios_scores_csv,
            bios_max_rows=args.bios_max_rows,
        )
        cmds.append((surrogate, strategy, seed, cmd))
    
    # Summary
    print(f"=== Surrogate Model Ablations ===")
    print(f"Dataset: {args.dataset}")
    print(f"Surrogates: {args.surrogates}")
    print(f"Strategies: {args.strategies}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total runs: {len(cmds)}")
    print()
    
    for surrogate in args.surrogates:
        print(f"  {surrogate}: {SURROGATES[surrogate]}")
    print()
    
    if args.dry_run:
        print("DRY RUN - Commands:")
        for i, (surr, strat, sd, c) in enumerate(cmds, 1):
            print(f"\n[{i}/{len(cmds)}] {surr} / {strat} / seed={sd}")
            print(" ".join(map(str, c)))
        return
    
    # Execute
    for i, (surr, strat, sd, c) in enumerate(cmds, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(cmds)}] Surrogate={surr}, Strategy={strat}, Seed={sd}")
        print(f"{'='*60}\n")
        
        try:
            subprocess.run(c, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n!!! Failed with exit code {e.returncode} !!!")
            print("Continue? (y/n): ", end="")
            if input().lower() != 'y':
                sys.exit(1)


if __name__ == "__main__":
    main()
