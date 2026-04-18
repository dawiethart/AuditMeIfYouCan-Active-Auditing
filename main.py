#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import glob
import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import wandb

from audit_run import AuditRunner
from config import AuditConfig

from data_loader import (
    load_jigsaw,
    load_bios,
    load_sbic_and_train_api_df,
    PROF_NAME_TO_ID,  # maps profession_name -> id
)

from surrogate_model import (
    compute_group_auc_difference,
    load_lora_bert_surrogate,
)

from blackboxes.blackbox_api_BERT import BlackBoxAPI
from blackboxes.blackbox_api_perspective_offline import OfflinePerspectiveBlackBox
from blackboxes.blackbox_api_bias_in_bios import BiasInBiosBlackBox


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_global_seeds(seed: int) -> int:
    seed = int(seed or 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def load_dataset(
    dataset_name: str,
    *,
    bios_target: str,
    bios_max_rows: Optional[int],
    seed: int,
) -> pd.DataFrame:
    dataset_name = dataset_name.lower()
    if dataset_name == "jigsaw":
        df = load_jigsaw()
    elif dataset_name == "bios":
        df = load_bios(target_profession=bios_target, max_rows=bios_max_rows, seed=seed)
    else:
        raise ValueError(f"Unknown --dataset: {dataset_name}. Choose from jigsaw|bios.")

    # required columns (bios may contain extra columns like "profession" which we keep)
    for col in ["id", "text", "group", "true_label"]:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain column '{col}'.")

    df["id"] = df["id"].astype(int)
    df["group"] = df["group"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    return df.reset_index(drop=True)


def make_blackbox(
    blackbox_name: str,
    dataset_name: str,
    *,
    dataset_df: pd.DataFrame,
    sbic_path: Optional[str],
    flip_probs: Optional[dict],
    perspective_csv_glob: Optional[str],
    bios_scores_csv: Optional[str],
    bios_target_label: str,
) -> Tuple[Any, Dict[str, Any], Optional[set]]:
    """
    Returns:
      (black_box_api_fn(ids)->scores, meta_dict, allowed_ids_set_or_None)
    """
    blackbox_name = blackbox_name.lower()
    dataset_name = dataset_name.lower()

    # id->text mapping for text-based blackboxes
    id_to_text = dict(zip(dataset_df["id"].astype(int).tolist(), dataset_df["text"].astype(str).tolist()))

    # --- HuggingFace classifier black-boxes ---
    if blackbox_name in {"hatebert", "bert"}:
        hf_name = "GroNLP/hateBERT" if blackbox_name == "hatebert" else "bert-base-uncased"
        bb = BlackBoxAPI(model_name_or_path=hf_name)

        if sbic_path is not None:
            bb = load_sbic_and_train_api_df(bb, path=sbic_path, flip_probs=flip_probs)

        def predict_scores(ids):
            ids = [int(i) for i in ids]
            texts = [id_to_text[i] for i in ids]
            return bb.predict_scores(texts)

        return predict_scores, {"blackbox_name": blackbox_name, "hf_name": hf_name, "sbic_path": sbic_path}, None

    # --- Offline Perspective ---
    if blackbox_name == "perspective_offline":
        if dataset_name != "jigsaw":
            raise ValueError("perspective_offline currently assumes Jigsaw IDs match your CSVs.")
        if not perspective_csv_glob:
            raise ValueError("Provide --perspective_csv_glob, e.g. 'data/perspective_jigsaw_group_*.csv'")

        csv_paths = sorted(glob.glob(perspective_csv_glob))
        if len(csv_paths) == 0:
            raise FileNotFoundError(f"No CSVs found for glob: {perspective_csv_glob}")

        offline_bb = OfflinePerspectiveBlackBox.from_csvs(csv_paths, strict=True)

        def predict_scores(ids):
            ids = [int(i) for i in ids]
            return offline_bb.predict_scores(ids=ids)

        return predict_scores, {"blackbox_name": "perspective_offline", "offline_csvs": len(csv_paths)}, None

    # --- Bias-in-Bios CSV blackbox (single-target fallback; multi-target handled in main()) ---
    if blackbox_name == "bios_csv":
        if dataset_name != "bios":
            raise ValueError("bios_csv blackbox only valid for --dataset bios")
        if not bios_scores_csv:
            raise ValueError("For --blackbox bios_csv, set --bios_scores_csv PATH")
        bios_target_label = str(bios_target_label)

        bb = BiasInBiosBlackBox(bios_scores_csv, verbose=True)
        allowed_ids_int = set(int(s[2:]) for s in bb.ids)

        def predict_scores(ids):
            sids = [f"ID{int(i)}" for i in ids]
            probs = bb.query_distribution(sids)
            if probs.ndim == 1:
                probs = probs[None, :]
            try:
                j = bb.labels.index(bios_target_label)
            except ValueError as e:
                raise ValueError(
                    f"bios_target_label='{bios_target_label}' not in CSV score cols. "
                    f"Available: {bb.labels}"
                ) from e
            return probs[:, j].astype(float)

        meta = {
            "blackbox_name": "bios_csv",
            "bios_scores_csv": bios_scores_csv,
            "bios_target_label": bios_target_label,
            "bb_rows_kept": bb.meta.n_rows_kept,
            "bb_score_cols": bb.meta.n_score_cols,
        }
        return predict_scores, meta, allowed_ids_int

    raise ValueError(
        f"Unknown --blackbox: {blackbox_name}. "
        f"Choose from hatebert|bert|perspective_offline|bios_csv."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AuditMeIfYouCan: Active Fairness Auditing for ML Models"
    )

    # === Dataset and Model Selection ===
    p.add_argument(
        "--dataset",
        type=str,
        default="jigsaw",
        choices=["jigsaw", "bios"],
        help="Dataset to audit: jigsaw (CivilComments) or bios (Bias-in-Bios)",
    )
    p.add_argument(
        "--blackbox",
        type=str,
        default="hatebert",
        choices=["hatebert", "bert", "perspective_offline", "bios_csv"],
        help="Black-box model type",
    )
    p.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Surrogate model name (HuggingFace identifier)",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="bo",
        choices=["stratified", "random", "disagreement", "expected_width_reduction", "bo", "bo_hybrid"],
        help="Selection strategy for active auditing",
    )

    # === Audit Loop Hyperparameters ===
    p.add_argument("--size_T", type=int, default=64, help="Top-k batch size (queries per round)")
    p.add_argument("--iterations", type=int, default=50, help="Total audit rounds")
    p.add_argument("--epochs_sur", type=int, default=4, help="Surrogate training epochs per round")
    p.add_argument("--epochs_opt", type=int, default=3, help="C-ERM optimization steps")
    p.add_argument("--batch_size", type=int, default=256, help="C-ERM batch size")
    p.add_argument("--lambda_penalty", type=float, default=1e-2, help="Constraint tolerance")
    p.add_argument("--epsilon", type=float, default=1e-2, help="Target precision / stopping threshold")

    # === Logging & Reproducibility ===
    p.add_argument("--title", type=str, default="run", help="Run title/name for logging")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results (can override OUTPUT_DIR env var)",
    )
    p.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (defaults to env var WANDB_PROJECT or 'audit-repo')",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name (defaults to env var WANDB_ENTITY or 'anonymous')",
    )

    # === Dataset-specific Options ===
    # Jigsaw/SBIC options
    p.add_argument("--sbic_path", type=str, default=None, help="Path to SBIC CSV")
    p.add_argument("--flip_black", type=float, default=None, help="Flip probability for black group")
    p.add_argument("--flip_white", type=float, default=None, help="Flip probability for white group")
    p.add_argument("--perspective_csv_glob", type=str, default=None, help="Glob pattern for Perspective API scores")

    # Bias-in-Bios options
    p.add_argument(
        "--bios_scores_csv",
        type=str,
        default=None,
        help="Path to precomputed Bias-in-Bios scores (blackbox_bios.csv)",
    )
    p.add_argument(
        "--bios_target_label",
        type=str,
        default="professor",
        help="One-vs-rest target label for single-target mode",
    )
    p.add_argument(
        "--bios_targets",
        nargs="*",
        default=None,
        help="Run multiple one-vs-rest targets sequentially",
    )
    p.add_argument(
        "--bios_max_rows",
        type=int,
        default=None,
        help="Optional subsample size for Bias-in-Bios (for quick debugging)",
    )
    p.add_argument(
        "--bios_skip_invalid_targets",
        action="store_true",
        help="Skip targets where either group lacks positives/negatives for AUC",
    )
    p.add_argument(
        "--bios_default_max_rows",
        type=int,
        default=20000,
        help="Safety default for --bios_max_rows if not specified (avoids accidental 391k row runs)",
    )

    # === Bayesian Optimization Parameters ===
    p.add_argument("--k_batch", type=int, default=8, help="Batch size for selection")
    p.add_argument("--bo_beta", type=float, default=1.0, help="UCB exploration parameter")
    p.add_argument("--reg_alpha", type=float, default=1.0, help="Distribution matching penalty")
    p.add_argument(
        "--bo_acq",
        type=str,
        default="ucb",
        choices=["ucb", "ei"],
        help="Acquisition function (ucb=Upper Confidence Bound, ei=Expected Improvement)",
    )
    p.add_argument("--bo_diversity_gamma", type=float, default=0.2, help="Diversity penalty weight")

    return p


def _check_auc_defined(df: pd.DataFrame) -> bool:
    """AUC per group requires each group to have both labels 0/1 present."""
    for g in [0, 1]:
        sub = df[df["group"] == g]
        if sub["true_label"].nunique() < 2:
            return False
    return True


def main():
    args = build_parser().parse_args()
    seed = set_global_seeds(args.seed)

    # --- bios safety default: avoids accidental full 391k runs ---
    if args.dataset.lower() == "bios" and args.bios_max_rows is None:
        args.bios_max_rows = int(args.bios_default_max_rows)
        print(f"[MAIN] --bios_max_rows not set; defaulting to {args.bios_max_rows} for safety.")
    elif args.dataset.lower() == "bios":
        print(f"[MAIN] Using --bios_max_rows={args.bios_max_rows}")

    # determine targets
    targets: List[str]
    if args.dataset.lower() == "bios" and args.bios_targets and len(args.bios_targets) > 0:
        targets = [str(t) for t in args.bios_targets]
    else:
        targets = [str(args.bios_target_label)]

    # Load base dataset ONCE
    base_target_for_load = targets[0] if args.dataset.lower() == "bios" else args.bios_target_label
    dataset_base = load_dataset(
        args.dataset,
        bios_target=base_target_for_load,
        bios_max_rows=args.bios_max_rows,
        seed=seed,
    )

    # Flip probs (if used)
    flip_probs = None
    if args.flip_black is not None and args.flip_white is not None:
        flip_probs = {"black": float(args.flip_black), "white": float(args.flip_white)}

    # Tokenizer used by AuditRunner as C-ERM tokenizer
    tokenizer_sur, _ = load_lora_bert_surrogate()

    # Multi-target bios_csv: preload the big CSV ONCE and reuse across targets
    bios_bb: Optional[BiasInBiosBlackBox] = None
    bios_allowed_ids: Optional[set[int]] = None
    if args.blackbox.lower() == "bios_csv":
        if not args.bios_scores_csv:
            raise ValueError("--blackbox bios_csv requires --bios_scores_csv PATH")
        bios_bb = BiasInBiosBlackBox(args.bios_scores_csv, verbose=True)
        bios_allowed_ids = set(int(s[2:]) for s in bios_bb.ids)

        # filter base dataset once to BB coverage (huge speed/memory win)
        before = len(dataset_base)
        dataset_base = dataset_base[dataset_base["id"].isin(bios_allowed_ids)].reset_index(drop=True)
        after = len(dataset_base)
        print(f"[MAIN] (base) Filtered dataset to BB coverage: {before} -> {after}")

    results = []

    for t in targets:
        # Build target-specific dataset (fresh copy; AuditRunner mutates it)
        dataset_D = dataset_base.copy()

        # set one-vs-rest labels for this target
        if args.dataset.lower() == "bios":
            if "profession" not in dataset_D.columns:
                raise ValueError("Bias-in-Bios dataset must include 'profession' column (load_bios must return it).")
            if t not in PROF_NAME_TO_ID:
                raise ValueError(f"Unknown bios target '{t}'. Known: {sorted(PROF_NAME_TO_ID.keys())}")
            target_id = int(PROF_NAME_TO_ID[t])
            dataset_D["true_label"] = (dataset_D["profession"].astype(int) == target_id).astype(int)

            # ensure no stale bb_score if reusing frames
            if "bb_score" in dataset_D.columns:
                dataset_D = dataset_D.drop(columns=["bb_score"])

        # Build blackbox (target-specific)
        if args.blackbox.lower() == "bios_csv":
            assert bios_bb is not None

            if t not in bios_bb.labels:
                msg = f"[MAIN] Target '{t}' not in CSV score columns. Available: {bios_bb.labels}"
                if args.bios_skip_invalid_targets:
                    print(msg + " -> skipping")
                    continue
                raise ValueError(msg)

            col_idx = bios_bb.labels.index(t)

            def black_box_api_fn(ids, _col=col_idx):
                sids = [f"ID{int(i)}" for i in ids]
                probs = bios_bb.query_distribution(sids)
                if probs.ndim == 1:
                    probs = probs[None, :]
                return probs[:, _col].astype(float)

            bb_meta = {
                "blackbox_name": "bios_csv",
                "bios_scores_csv": args.bios_scores_csv,
                "bios_target_label": t,
                "bb_rows_kept": bios_bb.meta.n_rows_kept,
                "bb_score_cols": bios_bb.meta.n_score_cols,
            }

        else:
            black_box_api_fn, bb_meta, allowed_ids = make_blackbox(
                blackbox_name=args.blackbox,
                dataset_name=args.dataset,
                dataset_df=dataset_D,
                sbic_path=args.sbic_path,
                flip_probs=flip_probs,
                perspective_csv_glob=args.perspective_csv_glob,
                bios_scores_csv=args.bios_scores_csv,
                bios_target_label=t,
            )
            if allowed_ids is not None:
                before = len(dataset_D)
                dataset_D = dataset_D[dataset_D["id"].isin(allowed_ids)].reset_index(drop=True)
                after = len(dataset_D)
                print(f"[MAIN] Filtered dataset to BB coverage: {before} -> {after}")

        # Safety: ensure AUC is defined per group for this target
        if args.dataset.lower() == "bios":
            if not _check_auc_defined(dataset_D):
                msg = (
                    f"[MAIN] ({t}) AUC undefined for at least one group "
                    f"(needs both labels 0/1 within each group)."
                )
                if args.bios_skip_invalid_targets:
                    print(msg + " -> skipping")
                    continue
                raise ValueError(msg)

        # Config
        config = AuditConfig(
            model=args.model,
            size_T=args.size_T,
            iterations=args.iterations,
            epochs_sur=args.epochs_sur,
            epochs_opt=args.epochs_opt,
            batch_size=args.batch_size,
            lambda_penalty=args.lambda_penalty,
            epsilon=args.epsilon,
            strategy=args.strategy,

            k_batch=args.k_batch,
            bo_beta=args.bo_beta,
            reg_alpha=args.reg_alpha,
            bo_acq=args.bo_acq,
            bo_diversity_gamma=args.bo_diversity_gamma,

            seed=seed,
            dataset=args.dataset,
            blackbox=args.blackbox,
            
            # Logging & output
            output_dir=args.output_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            title=args.title,
        )

        # W&B run per target (clean for later averaging)
        group_name = f"{args.dataset}-{args.blackbox}-{args.strategy}"
        run_name = f"{args.title}-target{t}-{args.dataset}-{args.blackbox}-{args.strategy}-seed{seed}"

        # Initialize W&B only if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                group=group_name,
                config={
                    **config.__dict__,
                    **bb_meta,
                    "bios_max_rows": args.bios_max_rows,
                    "bios_targets": targets,
                },
                job_type="strategy_comparison",
                reinit=True,
            )
        else:
            # Disable W&B logging
            os.environ["WANDB_MODE"] = "disabled"
            wandb.init(mode="disabled")

        try:
            runner = AuditRunner(
                dataset_D=dataset_D,
                black_box_api_fn=black_box_api_fn,
                compute_group_auc_diff_fn=compute_group_auc_difference,
                tokenizer=tokenizer_sur,
                config=config,
            )
            out = runner.run()
            results.append(
                {
                    "target": t,
                    "summary": out.get("summary", {}),
                    "delta_bb": out.get("delta_bb"),
                }
            )
        finally:
            wandb.finish()

    # Optional: quick aggregate across targets
    if len(results) > 1:
        deltas = [abs(r["delta_bb"]) for r in results if r.get("delta_bb") is not None]
        print("\n" + "=" * 80)
        print(f"[MAIN] Completed {len(results)}/{len(targets)} targets.")
        if deltas:
            print(f"[MAIN] Mean |ΔAUC_bb| over targets: {float(np.mean(deltas)):.6f}")
            print(f"[MAIN] Std  |ΔAUC_bb| over targets: {float(np.std(deltas)):.6f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
