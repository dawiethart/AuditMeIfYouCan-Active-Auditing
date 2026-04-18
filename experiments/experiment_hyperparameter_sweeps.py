#!/usr/bin/env python3
"""
Hyperparameter Sweeps: Systematic ablations of key BAFA parameters.

Based on paper Appendix A.4.3:
- Table 5: epochs_opt ∈ {3, 6, 8, 10}
- Table 6: batch sizes (k and B_cerm)
- BO parameters: beta, diversity_gamma

Usage:
  # Run all hyperparameter sweeps
  python experiment_hyperparameter_sweeps.py --sweep all
  
  # Run specific sweep
  python experiment_hyperparameter_sweeps.py --sweep epochs_opt
  python experiment_hyperparameter_sweeps.py --sweep batch_size
  python experiment_hyperparameter_sweeps.py --sweep bo_params
"""
import argparse
import subprocess
import sys
from itertools import product

# Sweep configurations from paper
SWEEPS = {
    "epochs_opt": {
        "description": "C-ERM optimization epochs (Table 5)",
        "values": [3, 6, 8, 10],
        "param": "epochs_opt",
    },
    "active_batch": {
        "description": "Active batch size k (queries/round, Table 6)",
        "values": [8, 16, 32],
        "param": "size_T",
    },
    "cerm_batch": {
        "description": "C-ERM batch size B_cerm (Table 6)",
        "values": [256, 512, 1024, 2048],
        "param": "batch_size",
    },
    "bo_beta": {
        "description": "BO exploration parameter β",
        "values": [0.5, 1.0, 2.0],
        "param": "bo_beta",
    },
    "bo_diversity": {
        "description": "BO diversity penalty γ",
        "values": [0.0, 0.1, 0.2, 0.5],
        "param": "bo_diversity_gamma",
    },
    "reg_alpha": {
        "description": "Distribution regularization α",
        "values": [0.5, 1.0, 2.0, 4.0],
        "param": "reg_alpha",
    },
}

# Use fewer seeds for sweeps (5 per configuration)
DEFAULT_SEEDS = list(range(5))


def build_cmd(
    python_exe: str,
    dataset: str,
    blackbox: str,
    strategy: str,
    seed: int,
    sweep_name: str,
    sweep_value: float | int,
    title_prefix: str,
    bios_scores_csv: str | None = None,
):
    """Build command with one parameter varied."""
    
    # Baseline configuration (Table 7)
    config = {
        "size_T": 16,
        "iterations": 75,
        "epochs_sur": 4,
        "epochs_opt": 10 if dataset.lower() == "jigsaw" else 8,
        "batch_size": 512,
        "lambda_penalty": "1e-2",
        "epsilon": "1e-2",
        "reg_alpha": 2.0,
        "bo_beta": 1.0,
        "bo_diversity_gamma": 0.2,
        "k_batch": 16,
    }
    
    # Override the swept parameter
    sweep_config = SWEEPS[sweep_name]
    param_name = sweep_config["param"]
    config[param_name] = sweep_value
    
    # Ensure k_batch matches size_T
    if param_name == "size_T":
        config["k_batch"] = sweep_value
    
    # Build title with sweep info
    title = f"{title_prefix}_{sweep_name}_{param_name}{sweep_value}_{strategy}_seed{seed}"
    
    cmd = [
        python_exe, "main.py",
        "--dataset", dataset,
        "--blackbox", blackbox,
        "--strategy", strategy,
        "--model", "bert-base-uncased",
        
        # All hyperparameters
        "--size_T", str(config["size_T"]),
        "--iterations", str(config["iterations"]),
        "--epochs_sur", str(config["epochs_sur"]),
        "--epochs_opt", str(config["epochs_opt"]),
        "--batch_size", str(config["batch_size"]),
        "--lambda_penalty", str(config["lambda_penalty"]),
        "--epsilon", str(config["epsilon"]),
        "--reg_alpha", str(config["reg_alpha"]),
        "--k_batch", str(config["k_batch"]),
        
        # BO parameters (used only if strategy=bo)
        "--bo_beta", str(config["bo_beta"]),
        "--bo_diversity_gamma", str(config["bo_diversity_gamma"]),
        "--bo_acq", "ucb",
        
        "--seed", str(seed),
        "--title", title,
    ]
    
    # Bias-in-Bios
    if dataset.lower() == "bios" and bios_scores_csv:
        cmd += ["--bios_scores_csv", bios_scores_csv]
    
    return cmd


def main():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep experiments"
    )
    
    p.add_argument(
        "--sweep",
        type=str,
        choices=list(SWEEPS.keys()) + ["all"],
        default="all",
        help="Which parameter to sweep"
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
        choices=["hatebert", "bert", "bios_csv"]
    )
    p.add_argument(
        "--strategies",
        nargs="*",
        choices=["bo", "disagreement"],
        default=["bo", "disagreement"],
        help="Which strategies to test"
    )
    p.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Seeds to use (default: 0-4)"
    )
    p.add_argument(
        "--bios_scores_csv",
        type=str
    )
    p.add_argument(
        "--title_prefix",
        type=str,
        default="hyperparam_sweep"
    )
    p.add_argument(
        "--dry_run",
        action="store_true"
    )
    
    args = p.parse_args()
    
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS
    python_exe = sys.executable
    
    # Determine which sweeps to run
    if args.sweep == "all":
        sweeps_to_run = list(SWEEPS.keys())
    else:
        sweeps_to_run = [args.sweep]
    
    # Filter sweeps by strategy compatibility
    if "bo" not in args.strategies:
        # Skip BO-specific sweeps if not running BO
        sweeps_to_run = [
            s for s in sweeps_to_run 
            if s not in ["bo_beta", "bo_diversity"]
        ]
    
    # Build all commands
    all_cmds = []
    for sweep_name in sweeps_to_run:
        sweep_config = SWEEPS[sweep_name]
        
        # Determine applicable strategies for this sweep
        if sweep_name in ["bo_beta", "bo_diversity"]:
            applicable_strategies = ["bo"]
        else:
            applicable_strategies = args.strategies
        
        for value, strategy, seed in product(
            sweep_config["values"],
            applicable_strategies,
            seeds
        ):
            cmd = build_cmd(
                python_exe=python_exe,
                dataset=args.dataset,
                blackbox=args.blackbox,
                strategy=strategy,
                seed=seed,
                sweep_name=sweep_name,
                sweep_value=value,
                title_prefix=args.title_prefix,
                bios_scores_csv=args.bios_scores_csv,
            )
            all_cmds.append((sweep_name, value, strategy, seed, cmd))
    
    # Summary
    print(f"=== Hyperparameter Sweeps ===")
    print(f"Dataset: {args.dataset}")
    print(f"Sweeps to run: {sweeps_to_run}")
    print(f"Seeds: {len(seeds)}")
    print(f"Total runs: {len(all_cmds)}")
    print()
    
    for sweep_name in sweeps_to_run:
        sweep_config = SWEEPS[sweep_name]
        print(f"  {sweep_name}:")
        print(f"    {sweep_config['description']}")
        print(f"    Values: {sweep_config['values']}")
        print()
    
    if args.dry_run:
        print("DRY RUN - Sample commands:")
        for i, (sweep, val, strat, sd, c) in enumerate(all_cmds[:3], 1):
            print(f"\n[{i}] {sweep}={val}, {strat}, seed={sd}")
            print(" ".join(map(str, c)))
        print(f"\n... and {len(all_cmds) - 3} more")
        return
    
    # Execute
    for i, (sweep, val, strat, sd, c) in enumerate(all_cmds, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(all_cmds)}] Sweep={sweep}, Value={val}, Strategy={strat}, Seed={sd}")
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
