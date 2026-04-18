# audit_run.py
from __future__ import annotations

import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score

from optimization import eval_h
from utils import (
    df_map,
    fresh_model,
    compute_group_auc_difference_from_scores,
    create_stratified_batches,
    sample_stratified_fixed_size,
    free_model,
)

from selection import AuditSelector
import surrogate_model


class AuditRunner:
    def __init__(
        self,
        dataset_D: pd.DataFrame,
        black_box_api_fn,
        compute_group_auc_diff_fn,
        tokenizer,
        config,
    ):
        self.D = dataset_D.copy().reset_index(drop=True)
        self.api = black_box_api_fn
        self.compute_auc_diff = compute_group_auc_diff_fn
        self.tokenizer = tokenizer  # C-ERM tokenizer
        self.config = config

        self.wandb_prefix = getattr(self.config, "wandb_prefix", "audit")
        self.seed = int(getattr(self.config, "seed", 0))
        self.strategy = str(getattr(self.config, "strategy", "bo")).lower()

        # surrogate settings
        self.use_surrogate = bool(getattr(self.config, "use_surrogate", True))
        self.surrogate_update_every = int(getattr(self.config, "surrogate_update_every", 1))
        self.surrogate_epochs = int(getattr(self.config, "surrogate_epochs", 1))
        self.surrogate_batch_size = int(getattr(self.config, "surrogate_batch_size", 16))
        self.surrogate_device = str(
            getattr(
                self.config,
                "surrogate_device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        # K-Size how many samples we take each step
        self.k_batch = int(getattr(self.config, "k_batch", 8)) 
        
        # regularizer
        self.reg_alpha=float(getattr(self.config, "reg_alpha", 1.0)) 

        # BO settings
        self.bo_beta = float(getattr(self.config, "bo_beta", 1.0))
        self.bo_min_points = int(getattr(self.config, "bo_min_points", 12))

        if "group" not in self.D.columns:
            raise ValueError("dataset_D must contain a 'group' column")
        if "text" not in self.D.columns:
            raise ValueError("dataset_D must contain a 'text' column")
        if "true_label" not in self.D.columns:
            raise ValueError("dataset_D must contain a 'true_label' column")

        # Ensure id column
        if "id" not in self.D.columns:
            self.D["id"] = np.arange(len(self.D), dtype=int)
        # keep ids stable & castable
        self.D["id"] = self.D["id"].astype(int)

        # -----------------------------
        # stratified batches for C-ERM
        # -----------------------------
        batch_size_cerm = int(getattr(self.config, "batch_size", 128))
        self.D = create_stratified_batches(self.D, batch_size=batch_size_cerm).reset_index(drop=True)
        print(f"[INFO] Using stratified-batch version of D with {len(self.D)} examples.")

        # -----------------------------
        # group mapping (keep stable!)
        # -----------------------------
        if not np.issubdtype(self.D["group"].dtype, np.integer):
            mapping = {"white": 0, "black": 1}
            missing = set(self.D["group"].unique()) - set(mapping.keys())
            if missing:
                raise ValueError(f"Group labels {missing} not covered by mapping {mapping}")
            self.D["group"] = self.D["group"].map(mapping).astype(int)
            self.group_mapping = mapping
            print(f"[INFO] Using explicit group mapping: {self.group_mapping}")
        else:
            self.group_mapping = None

        # id -> row index for score lookup into arrays over D
        self.id_to_idx = {int(_id): int(i) for i, _id in enumerate(self.D["id"].astype(int).tolist())}

        # Precompute bb_score on full D (offline simulation)
        if "bb_score" not in self.D.columns:
            print("[INFO] Computing black-box scores for full D (prefer id)...")
            self.D["bb_score"] = self._api_scores(self.D)

        # Ground truth ΔAUC(D) from bb_score
        print("[INFO] Computing ground truth ΔAUC(D) from full D...")
        group_values = sorted(self.D["group"].unique().tolist())
        auc_by_group: Dict[int, float] = {}

        for g in group_values:
            sub = self.D[self.D["group"] == g]
            if sub["true_label"].nunique() < 2:
                auc_by_group[g] = np.nan
            else:
                auc_by_group[g] = roc_auc_score(
                    sub["true_label"].astype("float64").to_numpy(),
                    sub["bb_score"].to_numpy(dtype=float),
                )

        # assumes exactly two groups 0/1
        self.delta_auc_blackbox = float(auc_by_group[0] - auc_by_group[1])
        self.auc_by_group = auc_by_group

        print(f"[INFO] ΔAUC_bb(D) = {self.delta_auc_blackbox:.4f}")
        print(f"       AUC(group=0)={auc_by_group[0]:.4f}, AUC(group=1)={auc_by_group[1]:.4f}")

        self._log_wandb(
            {
                "strategy": self.strategy,
                "delta_bb": float(self.delta_auc_blackbox),
                "delta_bb_abs": float(abs(self.delta_auc_blackbox)),
                "auc_group0": float(auc_by_group[0]),
                "auc_group1": float(auc_by_group[1]),
                "dataset_size": int(len(self.D)),
                "use_surrogate": int(self.use_surrogate),
                "surrogate_device": self.surrogate_device,
            },
            step=0,
            scope="init",
        )

        # Tokenize full D for C-ERM (true labels; not bb_score)
        print("[INFO] Tokenizing full D for C-ERM...")
        _, self.df_D_mapped = df_map(self.D, self.tokenizer, surrogate=False)
        self.inputs_D = {
            "input_ids": torch.tensor(self.df_D_mapped["input_ids"]).long(),
            "attention_mask": torch.tensor(self.df_D_mapped["attention_mask"]).long(),
            "labels": torch.tensor(self.df_D_mapped["labels"]).long(),
        }

        # BO state
        self.bo_state: Dict[str, Any] = {"X": None, "y": None, "gp": None, "kernel": None}

        # -----------------------------
        # Surrogate init (LoRA-BERT)
        # -----------------------------
        self.surr_tokenizer = None
        self.surr_model = None
        self._T_embed_cache: Optional[np.ndarray] = None
        self._T_embed_cache_ids: Optional[np.ndarray] = None

        if self.use_surrogate:
            print("[INFO] Loading LoRA surrogate...")
            self.surr_tokenizer, base_model = surrogate_model.load_lora_bert_surrogate()
            try:
                base_model.config.output_hidden_states = True
            except Exception:
                pass
            self.surr_model = base_model.to(self.surrogate_device)

    # -----------------------------
    # black-box query helper (prefer id)
    # -----------------------------
    def _api_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Query BB scores for df. Prefer calling api(ids) (stable),
        fall back to api(texts) for legacy wrappers.
        """
        ids = df["id"].astype(int).tolist()
        try:
            out = self.api(ids)
            return np.asarray(out, dtype=float)
        except TypeError:
            out = self.api(df["text"].tolist())
            return np.asarray(out, dtype=float)

    # -----------------------------
    # logging helper
    # -----------------------------
    def _log_wandb(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        scope: Optional[str] = None
    ):
        if scope:
            metrics = {f"{self.wandb_prefix}/{scope}/{k}": v for k, v in metrics.items()}
        else:
            metrics = {f"{self.wandb_prefix}/{k}": v for k, v in metrics.items()}

        print(f"[W&B] step={step} {metrics}")
        if wandb.run is not None:
            wandb.log(metrics, step=step)

    # -----------------------------
    # delta abs from df (bb_score)
    # -----------------------------
    def _delta_auc_abs_from_df(self, df: pd.DataFrame) -> float:
        scores = df["bb_score"].to_numpy(dtype=float)
        labels = df["true_label"].astype(int).to_numpy()
        groups = df["group"].astype(int).to_numpy()

        m0 = groups == 0
        m1 = groups == 1
        if m0.sum() == 0 or m1.sum() == 0:
            return float("nan")
        if len(np.unique(labels[m0])) < 2 or len(np.unique(labels[m1])) < 2:
            return float("nan")

        auc0 = roc_auc_score(labels[m0], scores[m0])
        auc1 = roc_auc_score(labels[m1], scores[m1])
        return float(abs(auc0 - auc1))

    # -----------------------------
    # baselines for current budget
    # -----------------------------
    def _log_baselines_for_budget(self, budget: int, step: int):
        if budget <= 0:
            return
        rng = np.random.RandomState(self.seed)

        rand_S = self.D.sample(n=budget, replace=False, random_state=rng)
        strat_S = sample_stratified_fixed_size(
            df=self.D,
            n=budget,
            rng=rng,
            group_col="group",
            label_col="true_label",
        )

        delta_true_abs = float(abs(self.delta_auc_blackbox))
        delta_rand = self._delta_auc_abs_from_df(rand_S)
        delta_strat = self._delta_auc_abs_from_df(strat_S)

        self._log_wandb(
            {
                "budget": int(budget),
                "delta_true_abs": delta_true_abs,
                "delta_random_abs": float(delta_rand),
                "delta_stratified_abs": float(delta_strat),
                "err_random_abs": float(abs(delta_rand - delta_true_abs)) if not np.isnan(delta_rand) else float("nan"),
                "err_stratified_abs": float(abs(delta_strat - delta_true_abs)) if not np.isnan(delta_strat) else float("nan"),
            },
            step=step,
            scope="baselines",
        )

    # -----------------------------
    # C-ERM bounds for current T
    # -----------------------------
    def _compute_cerm_bounds(self, T: pd.DataFrame, step: int) -> Dict[str, Any]:
        constraint_dict = {str(int(row["id"])): float(row["bb_score"]) for _, row in T.iterrows()}
        _, df_T_mapped = df_map(T, self.tokenizer, surrogate=True)

        epochs_opt = int(getattr(self.config, "epochs_opt", 4))
        batch_size = int(getattr(self.config, "batch_size", 128))
        lambda_penalty = float(getattr(self.config, "lambda_penalty", 0.1))

        # upper bound
        scores_max_on_D, h_max = eval_h(
            base_model_factory=fresh_model,
            df_D=self.D,
            df_D_mapped=self.df_D_mapped,
            inputs_D=self.inputs_D,
            df_T_mapped=df_T_mapped,
            constraint_pred=constraint_dict,
            epochs_opt=epochs_opt,
            batch_size=batch_size,
            lambda_penalty=lambda_penalty,
            tokenizer=self.tokenizer,
            Maximize=True,
            compute_group_auc_diff_fn=self.compute_auc_diff,
        )

        # lower bound
        scores_min_on_D, h_min = eval_h(
            base_model_factory=fresh_model,
            df_D=self.D,
            df_D_mapped=self.df_D_mapped,
            inputs_D=self.inputs_D,
            df_T_mapped=df_T_mapped,
            constraint_pred=constraint_dict,
            epochs_opt=epochs_opt,
            batch_size=batch_size,
            lambda_penalty=lambda_penalty,
            tokenizer=self.tokenizer,
            Maximize=False,
            compute_group_auc_diff_fn=self.compute_auc_diff,
        )

        delta_max, _ = compute_group_auc_difference_from_scores(
            scores_max_on_D, self.D, group1=1, group2=0, device="cpu"
        )
        delta_min, _ = compute_group_auc_difference_from_scores(
            scores_min_on_D, self.D, group1=1, group2=0, device="cpu"
        )
        delta_max = float(delta_max)
        delta_min = float(delta_min)
        width = float(delta_max - delta_min)

        self._log_wandb(
            {
                "T_size": int(len(T)),
                "delta_max_true": delta_max,
                "delta_min_true": delta_min,
                "width": width,
            },
            step=step,
            scope="cerm",
        )

        return {
            "delta_max_true": delta_max,
            "delta_min_true": delta_min,
            "width": width,
            "scores_max": np.asarray(scores_max_on_D, dtype=float),
            "scores_min": np.asarray(scores_min_on_D, dtype=float),
            "h_max": h_max,
            "h_min": h_min,
        }

    # -----------------------------
    # score fn helper
    # -----------------------------
    def _make_score_fn(self, scores_array: np.ndarray):
        id_to_idx = self.id_to_idx

        def fn(U: pd.DataFrame) -> np.ndarray:
            ids = U["id"].astype(int).to_numpy()
            idxs = [id_to_idx[int(i)] for i in ids]
            return np.asarray(scores_array, dtype=float)[idxs]

        return fn

    def expected_width_fn(self, U: pd.DataFrame, T_cur: pd.DataFrame) -> np.ndarray:
        """Reusable expected-width scoring function that uses the latest C-ERM bounds.

        This replaces the previous inner-function closure so it can be reused
        safely across iterations. Requires that `self.last_bounds` has been
        set by `_compute_cerm_bounds` earlier in the loop.
        """
        bounds = getattr(self, "last_bounds", None)
        if bounds is None:
            raise ValueError("expected_width_fn requires self.last_bounds to be set before calling")

        scores_max = np.asarray(bounds["scores_max"], dtype=float)
        scores_min = np.asarray(bounds["scores_min"], dtype=float)
        h_max = bounds.get("h_max", None)

        ids = U["id"].astype(int).to_numpy()
        idxs = [self.id_to_idx[int(i)] for i in ids]
        p_max = np.asarray(scores_max, dtype=float)[idxs]
        p_min = np.asarray(scores_min, dtype=float)[idxs]

        # numeric safety: ensure probabilities in [0,1]
        p_max = np.clip(p_max, 0.0, 1.0)
        p_min = np.clip(p_min, 0.0, 1.0)

        width_loc = np.abs(p_max - p_min)
        unc = 0.5 * (p_max * (1.0 - p_max) + p_min * (1.0 - p_min))
        base = width_loc * unc

        grad = self._gradient_score_cpu(h_max, U, T_cur)
        grad = np.asarray(grad, dtype=float)
        nans_grad = int(np.isnan(grad).sum())
        infs_grad = int(np.isinf(grad).sum())
        if nans_grad or infs_grad:
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            self._log_wandb({"expected_width_grad_nan_count": nans_grad, "expected_width_grad_inf_count": infs_grad}, step=getattr(self, "last_step", None), scope="selector_debug")

        grad = np.clip(grad, 0.0, 10.0)
        score = base * (1.0 + grad)

        # final validation: finite + correct shape
        nans = int(np.isnan(score).sum())
        infs = int(np.isinf(score).sum())
        if nans or infs:
            score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
            self._log_wandb({"expected_width_nan_count": nans, "expected_width_inf_count": infs}, step=getattr(self, "last_step", None), scope="selector_debug")

        if score.shape[0] != len(U):
            raise ValueError("expected_width_fn must return len(U) scores.")

        return score.astype(float)

    # -----------------------------
    # gradient hook (keep stub for now)
    # -----------------------------
    def _gradient_score_cpu(self, h_model, U: pd.DataFrame, T_cur: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(U), dtype=float)

    # -----------------------------
    # --- Surrogate: train on T (bb_score) ---
    # -----------------------------
    def _train_or_update_surrogate(self, T: pd.DataFrame, step: int):
        if not self.use_surrogate or self.surr_model is None:
            return

        _, df_T_mapped = df_map(T, self.surr_tokenizer, surrogate=True)

        t0 = time.time()
        self.surr_model = surrogate_model.train_surrogate(
            self.surr_model,
            self.surr_tokenizer,
            df_T_mapped,
            self.surrogate_epochs,
            self.surrogate_batch_size,
        )
        self.surr_model.to(self.surrogate_device)

        self._log_wandb(
            {
                "surrogate_trained_on_T": int(len(T)),
                "surrogate_epochs": int(self.surrogate_epochs),
                "surrogate_batch_size": int(self.surrogate_batch_size),
                "surrogate_train_min": float((time.time() - t0) / 60.0),
            },
            step=step,
            scope="surrogate",
        )

        self._T_embed_cache = None
        self._T_embed_cache_ids = None

    # -----------------------------
    # Surrogate inference helpers
    # -----------------------------
    @torch.no_grad()
    def _surrogate_predict_and_embed(
        self,
        texts: list[str],
        max_len: int = 128,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.surr_model is None or self.surr_tokenizer is None:
            return np.zeros(len(texts), dtype=float), None

        self.surr_model.eval()
        probs_all = []
        emb_all = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.surr_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.surrogate_device) for k, v in enc.items()}

            out = self.surr_model(**enc)
            logits = out.logits.view(-1).float()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(probs)

            emb = None
            hs = getattr(out, "hidden_states", None)
            if hs is not None and len(hs) > 0:
                last = hs[-1]
                emb = last[:, 0, :]
                emb_all.append(emb.detach().cpu().numpy())

        probs_all = np.concatenate(probs_all, axis=0).astype(float)

        if len(emb_all) == 0:
            return probs_all, None
        emb_all = np.concatenate(emb_all, axis=0).astype(float)
        return probs_all, emb_all

    def _get_T_embeddings(self, T_cur: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.use_surrogate or self.surr_model is None:
            return None

        ids = T_cur["id"].astype(int).to_numpy()
        if self._T_embed_cache is not None and self._T_embed_cache_ids is not None:
            if np.array_equal(ids, self._T_embed_cache_ids):
                return self._T_embed_cache

        _, emb = self._surrogate_predict_and_embed(
            T_cur["text"].tolist(),
            max_len=int(getattr(self.config, "surrogate_max_len", 128)),
            batch_size=int(getattr(self.config, "surrogate_infer_batch_size", 32)),
        )
        self._T_embed_cache_ids = ids
        self._T_embed_cache = emb
        return emb

    # -----------------------------
    # surrogate features for BO
    # -----------------------------
    def _surrogate_feat_cpu(self, U: pd.DataFrame, T_cur: pd.DataFrame) -> np.ndarray:
        n = len(U)
        if (not self.use_surrogate) or (self.surr_model is None):
            return np.zeros((n, 1), dtype=float)

        max_len = int(getattr(self.config, "surrogate_max_len", 128))
        infer_bs = int(getattr(self.config, "surrogate_infer_batch_size", 32))

        p, embU = self._surrogate_predict_and_embed(
            U["text"].tolist(),
            max_len=max_len,
            batch_size=infer_bs,
        )
        p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        unc = p * (1.0 - p)

        if embU is None:
            feats = np.stack([unc, p], axis=1)
            return feats.astype(float)

        embT = self._get_T_embeddings(T_cur)
        if embT is None or len(embT) == 0:
            div = np.ones(n, dtype=float)
        else:
            U_norm = embU / (np.linalg.norm(embU, axis=1, keepdims=True) + 1e-8)
            T_norm = embT / (np.linalg.norm(embT, axis=1, keepdims=True) + 1e-8)
            sims = U_norm @ T_norm.T
            max_sim = np.max(sims, axis=1)
            div = 1.0 - max_sim
            div = np.clip(div, 0.0, 2.0)

        feats = np.stack([unc, div, p], axis=1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(float)

    # -----------------------------
    # run single-strategy BAFA loop
    # -----------------------------
    def run(self) -> Dict[str, Any]:
        size_T_init = int(getattr(self.config, "size_T", 32))
        size_T_init = min(size_T_init, len(self.D))

      
        target_width = float(getattr(self.config, "epsilon", 0.05))
        max_iters = int(getattr(self.config, "iterations", 50))

        print(f"[RUN] strategy={self.strategy} size_T_init={size_T_init} k_batch={self.k_batch} epsilon={target_width}")

        rng = np.random.RandomState(self.seed)
        T = sample_stratified_fixed_size(
            df=self.D,
            n=size_T_init,
            rng=rng,
            group_col="group",
            label_col="true_label",
        ).copy()

        if "bb_score" not in T.columns:
            T = T.merge(self.D[["id", "bb_score"]], on="id", how="left")

        # initial surrogate training on seed T
        if self.use_surrogate:
            self._train_or_update_surrogate(T, step=0)

        history = []
        prev_width_for_bo: Optional[float] = None
        last_bo_features: Optional[np.ndarray] = None

        for it in range(max_iters):
            step = it + 1
            print("\n" + "=" * 70)
            print(f"[Outer] Iteration {step}/{max_iters} |T|={len(T)}")
            print("=" * 70)

            t0 = time.time()
            bounds = self._compute_cerm_bounds(T, step=step)
            # expose for expected_width_fn and debugging
            self.last_bounds = bounds
            self.last_step = step

            W = float(bounds["width"])
            dmax = float(bounds["delta_max_true"])
            dmin = float(bounds["delta_min_true"])

            dmid = 0.5 * (dmax + dmin)
            err_mid = float(abs(dmid - self.delta_auc_blackbox))

            self._log_wandb(
                {
                    "iter": int(step),
                    "T_size": int(len(T)),
                    "delta_mid": float(dmid),
                    "err_mid": float(err_mid),
                    "width": float(W),
                    "target_width": float(target_width),
                },
                step=step,
                scope="outer",
            )

            # Always compare to baselines at this budget
            self._log_baselines_for_budget(budget=len(T), step=step)

            history.append(
                {
                    "iter": step,
                    "T_size": len(T),
                    "delta_max": dmax,
                    "delta_min": dmin,
                    "delta_mid": dmid,
                    "width": W,
                    "err_mid": err_mid,
                }
            )

            if W <= target_width:
                self._log_wandb({"stopped": 1, "stop_reason": 0}, step=step, scope="stop")
                print(f"[STOP] width {W:.4f} <= epsilon {target_width:.4f}")
                free_model(bounds["h_max"])
                free_model(bounds["h_min"])
                #break

            if len(T) >= len(self.D):
                self._log_wandb({"stopped": 1, "stop_reason": 1}, step=step, scope="stop")
                print("[STOP] All points queried.")
                free_model(bounds["h_max"])
                free_model(bounds["h_min"])
                break

            # ---- selection ----
            scores_max = bounds["scores_max"]
            scores_min = bounds["scores_min"]
            h_max = bounds["h_max"]
            h_min = bounds["h_min"]

            # base model score callables (validate outputs when called)
            raw_model_low = self._make_score_fn(scores_min)
            raw_model_up = self._make_score_fn(scores_max)

            def make_validated_model_fn(raw_fn, name: str):
                def wrapped(U: pd.DataFrame) -> np.ndarray:
                    out = np.asarray(raw_fn(U), dtype=float)
                    if out.shape[0] != len(U):
                        raise ValueError(f"{name} must return shape (len(U),)")
                    n_nans = int(np.isnan(out).sum())
                    n_infs = int(np.isinf(out).sum())
                    if n_nans or n_infs:
                        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                        self._log_wandb({f"{name}_nan_count": n_nans, f"{name}_inf_count": n_infs}, step=step, scope="selector_debug")
                    return out

                return wrapped

            model_low = make_validated_model_fn(raw_model_low, "model_low")
            model_up = make_validated_model_fn(raw_model_up, "model_up")

            def expected_width_fn(U: pd.DataFrame, T_cur: pd.DataFrame) -> np.ndarray:
                ids = U["id"].astype(int).to_numpy()
                idxs = [self.id_to_idx[int(i)] for i in ids]
                p_max = np.asarray(scores_max, dtype=float)[idxs]
                p_min = np.asarray(scores_min, dtype=float)[idxs]

                width_loc = np.abs(p_max - p_min)
                unc = 0.5 * (p_max * (1.0 - p_max) + p_min * (1.0 - p_min))
                base = width_loc * unc

                grad = self._gradient_score_cpu(h_max, U, T_cur)
                grad = np.nan_to_num(np.asarray(grad, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                grad = np.clip(grad, 0.0, 10.0)

                score = base * (1.0 + grad)
                return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

            # selector
            if self.strategy == "stratified":
                selector = AuditSelector(
                    strategy="stratified",
                    seed=self.seed,
                    group_col="group",
                    label_col="true_label",
                )
            elif self.strategy == "disagreement":
                selector = AuditSelector(
                    strategy="disagreement",
                    seed=self.seed,
                    group_col="group",
                    label_col="true_label",
                    model_low=model_low,
                    model_up=model_up,
                    reg_alpha=self.reg_alpha,
                )
            elif self.strategy == "expected_width_reduction":
                selector = AuditSelector(
                    strategy="expected_width_reduction",
                    seed=self.seed,
                    group_col="group",
                    label_col="true_label",
                    expected_width_fn=self.expected_width_fn,
                    reg_alpha=self.reg_alpha,
                )
            elif self.strategy == "bo":
                selector = AuditSelector(
                    strategy="bo",
                    seed=self.seed,
                    group_col="group",
                    label_col="true_label",
                    model_low=model_low,
                    model_up=model_up,
                    gradient_fn=lambda U_, T_: self._gradient_score_cpu(h_max, U_, T_),
                    surrogate_feat_fn=self._surrogate_feat_cpu,
                    bo_state=self.bo_state,
                    bo_beta=self.bo_beta,
                    bo_min_points=self.bo_min_points,
                    reg_alpha=self.reg_alpha,
                    bo_acq=getattr(self.config, "bo_acq", "ucb"),
                    bo_diversity_gamma=getattr(self.config, "bo_diversity_gamma", 0.2),
                )
            elif self.strategy == "bo_hybrid":
                selector = AuditSelector(
                    strategy="bo_hybrid",
                    seed=self.seed,
                    group_col="group",
                    label_col="true_label",
                    model_low=model_low,
                    model_up=model_up,
                    expected_width_fn=self.expected_width_fn,
                    gradient_fn=lambda U_, T_: self._gradient_score_cpu(h_max, U_, T_),
                    surrogate_feat_fn=self._surrogate_feat_cpu,
                    bo_state=self.bo_state,
                    bo_beta=self.bo_beta,
                    bo_min_points=self.bo_min_points,
                    reg_alpha=self.reg_alpha,
                    bo_acq=getattr(self.config, "bo_acq", "ucb"),
                    bo_diversity_gamma=getattr(self.config, "bo_diversity_gamma", 0.2),
                )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # select batch
            B = selector.select_next_batch(D=self.D, T=T, k=self.k_batch)

            # --- selector debug logging: info, distribution weights, combined score ---
            try:
                if B is None or len(B) == 0:
                    pass
                else:
                    queried_ids = set(T["id"].tolist())
                    U = self.D[~self.D["id"].isin(queried_ids)].copy()

                    info_vals = None
                    combined_vals = None

                    if self.strategy == "disagreement":
                        info_vals = np.abs(model_up(B) - model_low(B))

                    elif self.strategy == "expected_width_reduction":
                        info_vals = self.expected_width_fn(B, T)

                    elif self.strategy in {"bo", "bo_hybrid"}:
                        # prefer using selector-provided arrays aligned with U
                        mask = U["id"].isin(B["id"]).to_numpy()
                        acq_all = selector.last_batch_acq_raw_
                        comb_all = selector.last_batch_combined_scores_
                        if acq_all is not None and len(acq_all) == len(U):
                            info_vals = np.asarray(acq_all, dtype=float)[mask]
                        if comb_all is not None and len(comb_all) == len(U):
                            combined_vals = np.asarray(comb_all, dtype=float)[mask]

                        # fallback: recompute acquisition on B if we couldn't slice
                        if info_vals is None or len(info_vals) != len(B):
                            try:
                                Xb = selector._build_feature_matrix(B, T)
                                info_vals = selector._gp_ucb(Xb)
                            except Exception:
                                info_vals = np.zeros(len(B), dtype=float)

                    # ensure arrays are numpy and correct length
                    info_vals = np.asarray(info_vals, dtype=float)
                    if info_vals.shape[0] != len(B):
                        self._log_wandb({"selector_debug_info_len_mismatch": 1}, step=step, scope="selector_debug")
                        info_vals = np.resize(info_vals, (len(B),))

                    # distribution weights for selected batch (use selector helper)
                    try:
                        w_vals = selector._distribution_weights(self.D, T, B)
                    except Exception:
                        w_vals = np.ones(len(B), dtype=float)

                    w_vals = np.asarray(w_vals, dtype=float)
                    if w_vals.shape[0] != len(B):
                        self._log_wandb({"selector_debug_w_len_mismatch": 1}, step=step, scope="selector_debug")
                        w_vals = np.resize(w_vals, (len(B),))

                    # final score (use combined if available, else info)
                    score_vals = None
                    if combined_vals is not None:
                        combined_vals = np.asarray(combined_vals, dtype=float)
                        if combined_vals.shape[0] == len(B):
                            score_vals = combined_vals * w_vals
                        else:
                            score_vals = info_vals * w_vals
                    else:
                        score_vals = info_vals * w_vals

                    # compute stats and log
                    metrics = {
                        "selector_debug/batch_size": int(len(B)),
                        "selector_debug/info_mean": float(np.mean(info_vals)),
                        "selector_debug/info_max": float(np.max(info_vals)),
                        "selector_debug/w_mean": float(np.mean(w_vals)),
                        "selector_debug/w_max": float(np.max(w_vals)),
                        "selector_debug/score_mean": float(np.mean(score_vals)),
                        "selector_debug/score_max": float(np.max(score_vals)),
                    }
                    self._log_wandb(metrics, step=step, scope="selector_debug")
            except Exception as e:
                # ensure selection loop doesn't crash from logging
                print(f"[WARN] selector debug logging failed: {e}")

            # BO update label: improvement from previous width to current width
            if self.strategy in {"bo", "bo_hybrid"}:
                if prev_width_for_bo is not None and last_bo_features is not None and last_bo_features.shape[0] > 0:
                    improvement = (prev_width_for_bo - W) / float(last_bo_features.shape[0])
                    y_last = np.full(last_bo_features.shape[0], improvement, dtype=float)

                    if self.bo_state["X"] is None:
                        self.bo_state["X"] = last_bo_features
                        self.bo_state["y"] = y_last
                    else:
                        self.bo_state["X"] = np.vstack([self.bo_state["X"], last_bo_features])
                        self.bo_state["y"] = np.concatenate([self.bo_state["y"], y_last])

                    self._log_wandb(
                        {
                            "bo_train_size": int(len(self.bo_state["y"])),
                            "bo_last_improvement_per_point": float(improvement),
                        },
                        step=step,
                        scope="bo",
                    )

                last_bo_features = selector.last_batch_features_

            # --- BO logging: rho, acq type, diversity gamma, mean/max acq among selected ---
            if self.strategy in {"bo", "bo_hybrid"}:
                try:
                    rho = float(self.bo_state.get("last_rho", 1.0))
                except Exception:
                    rho = 1.0
                acq_type = getattr(self.config, "bo_acq", "ucb")
                diversity_gamma = float(getattr(self.config, "bo_diversity_gamma", 0.2))

                acq_vals = selector.last_batch_acq_raw_
                combined_vals = selector.last_batch_combined_scores_
                if combined_vals is None:
                    combined_vals = acq_vals

                if combined_vals is not None and len(combined_vals) > 0:
                    mean_acq = float(np.mean(combined_vals))
                    max_acq = float(np.max(combined_vals))
                else:
                    mean_acq = float("nan")
                    max_acq = float("nan")

                self._log_wandb(
                    {
                        "rho": rho,
                        "acq_type": acq_type,
                        "diversity_gamma": diversity_gamma,
                        "selected_mean_acq": mean_acq,
                        "selected_max_acq": max_acq,
                    },
                    step=step,
                    scope="bo",
                )

            # free C-ERM models
            free_model(h_max)
            free_model(h_min)

            if B is None or len(B) == 0:
                self._log_wandb({"stopped": 1, "stop_reason": 2}, step=step, scope="stop")
                print("[STOP] Selector returned empty batch.")
                break

            if "bb_score" not in B.columns:
                B = B.merge(self.D[["id", "bb_score"]], on="id", how="left")

            old_size = len(T)
            T = pd.concat([T, B], ignore_index=True).drop_duplicates("id").reset_index(drop=True)

            self._log_wandb(
                {
                    "T_size_before": int(old_size),
                    "T_size_after": int(len(T)),
                    "batch_size": int(len(B)),
                    "iter_duration_min": float((time.time() - t0) / 60.0),
                },
                step=step,
                scope="growth",
            )

            # update surrogate after growing T (optional schedule)
            if self.use_surrogate and (step % self.surrogate_update_every == 0):
                self._train_or_update_surrogate(T, step=step)

            prev_width_for_bo = W

        # Final summary
        final = history[-1] if len(history) > 0 else {}
        summary = {
            "strategy": self.strategy,
            "final_iter": int(final.get("iter", 0)),
            "final_T_size": int(final.get("T_size", 0)),
            "final_width": float(final.get("width", float("nan"))),
            "final_delta_mid": float(final.get("delta_mid", float("nan"))),
            "final_err_mid": float(final.get("err_mid", float("nan"))),
            "delta_bb": float(self.delta_auc_blackbox),
            "delta_bb_abs": float(abs(self.delta_auc_blackbox)),
        }
        self._log_wandb(summary, step=summary["final_iter"], scope="final")

        return {"history": history, "summary": summary, "delta_bb": self.delta_auc_blackbox}
