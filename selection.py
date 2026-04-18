# selection.py
from __future__ import annotations

from typing import Literal, Callable, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from utils import sample_stratified_fixed_size

# CPU-only BO (feature-space GP)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class AuditSelector:
    """
    Selection class for active auditing.

    Strategies
    ----------
    - "stratified": proportional stratified sampling over (group, true_label)
    - "random": simple random sampling without replacement
    - "disagreement": choose points with maximal |p_up - p_low|, with distribution regularization
    - "expected_width_reduction": choose points maximizing expected_width_fn, with distribution regularization
    - "bo": disagreement-anchored BO (GP-UCB on features) + distribution regularization + diversity
    - "bo_hybrid": disagreement+expected_width anchor + BO stabilizer + distribution regularization + diversity

    Core design principle
    ---------------------
    BO should *not* override the core informativeness (disagreement/expected-width).
    It should act as a stabilizer/explorer: small mixing weight that ramps up over time.

    Important
    ---------
    This file assumes the *runner* updates bo_state["X"] and bo_state["y"] over time.
    We refit the GP whenever the BO dataset size changes (fixes "stale GP" behavior).
    """

    def __init__(
        self,
        strategy: Literal[
            "stratified",
            "random",
            "disagreement",
            "expected_width_reduction",
            "bo",
            "bo_hybrid",
        ] = "stratified",
        seed: int = 0,
        group_col: str = "group",
        label_col: str = "true_label",
        model_low: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        model_up: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        expected_width_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = None,
        gradient_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = None,
        surrogate_feat_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = None,
        bo_state: Optional[Dict[str, Any]] = None,
        bo_beta: float = 1.0,
        bo_min_points: int = 12,
        bo_acq: str = "ucb",
        bo_diversity_gamma: float = 0.2,
        # distribution regularization
        reg_alpha: float = 0.5,
        reg_cap: float = 5.0,
        reg_eps: float = 1e-6,
        reg_warmup: int = 32,
        reg_ramp: int = 32,
        # BO mixing schedule: keep BO as stabilizer, disagreement as anchor
        bo_mix_max: float = 0.35,     # max fraction of BO signal in final score
        bo_mix_warmup: int = 32,      # rounds with BO mixing = 0
        bo_mix_ramp: int = 32,        # linear ramp length to bo_mix_max
        # BO candidate restriction (optional safety): restrict BO to high-disagreement region
        bo_restrict_quantile: float = 0.0,  # 0 disables, e.g. 0.8 keeps top-20% disagreement
        # numerical safety
        eps: float = 1e-8,
    ):
        self.strategy = strategy
        self.rng = np.random.RandomState(seed)
        self.group_col = group_col
        self.label_col = label_col

        self.model_low = model_low
        self.model_up = model_up
        self.expected_width_fn = expected_width_fn
        self.gradient_fn = gradient_fn
        self.surrogate_feat_fn = surrogate_feat_fn

        # Distribution regularization
        self.reg_alpha = float(reg_alpha)
        self.reg_cap = float(reg_cap)
        self.reg_eps = float(reg_eps)
        self.reg_warmup = int(reg_warmup)
        self.reg_ramp = int(reg_ramp)

        # BO
        self.bo_state = bo_state or {}
        self.bo_beta = float(bo_beta)
        self.bo_min_points = int(bo_min_points)
        self.bo_acq = str(bo_acq).lower()
        self.bo_diversity_gamma = float(bo_diversity_gamma)

        # BO mixing (stability)
        self.bo_mix_max = float(bo_mix_max)
        self.bo_mix_warmup = int(bo_mix_warmup)
        self.bo_mix_ramp = int(bo_mix_ramp)
        self.bo_restrict_quantile = float(bo_restrict_quantile)

        self.eps = float(eps)

        # Logging hooks (optional)
        self.last_batch_acq_raw_ = None            # acq over U (len(U),)
        self.last_batch_info_raw_ = None           # base info over U (len(U),)
        self.last_batch_combined_scores_ = None    # final score over U (len(U),)
        self.last_batch_features_ = None           # selected X (k, d)
        self.last_batch_ids_ = None                # selected ids (k,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _zscore(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = float(np.mean(x)) if len(x) else 0.0
        sd = float(np.std(x)) if len(x) else 1.0
        return (x - mu) / (sd + self.eps)

    def _sigmoid01(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # Regularization schedule (distribution)
    # ------------------------------------------------------------------
    def _reg_strength(self, T: pd.DataFrame) -> float:
        t = len(T)
        if t <= self.reg_warmup:
            return 0.0
        if t >= self.reg_warmup + self.reg_ramp:
            return self.reg_alpha
        frac = (t - self.reg_warmup) / float(self.reg_ramp)
        return frac * self.reg_alpha

    # ------------------------------------------------------------------
    # BO mixing schedule (informativeness anchor vs BO stabilizer)
    # ------------------------------------------------------------------
    def _bo_mix_lambda(self, T: pd.DataFrame) -> float:
        t = len(T)
        if t <= self.bo_mix_warmup:
            return 0.0
        if t >= self.bo_mix_warmup + self.bo_mix_ramp:
            return self.bo_mix_max
        frac = (t - self.bo_mix_warmup) / float(self.bo_mix_ramp)
        return frac * self.bo_mix_max

    # ------------------------------------------------------------------
    # Distribution weights
    # ------------------------------------------------------------------
    def _distribution_weights(self, D: pd.DataFrame, T: pd.DataFrame, U: pd.DataFrame) -> np.ndarray:
        if len(D) == 0 or len(U) == 0:
            return np.ones(len(U), dtype=float)

        gcol, ycol = self.group_col, self.label_col

        counts_D = D.groupby([gcol, ycol]).size()
        p_D = counts_D / float(len(D))

        if len(T) > 0:
            counts_T = T.groupby([gcol, ycol]).size()
            p_T = counts_T / float(len(T))
        else:
            p_T = pd.Series(0.0, index=p_D.index)

        keys = U[[gcol, ycol]].copy()
        tmp = (
            keys.merge(p_D.rename("pD"), on=[gcol, ycol], how="left")
                .merge(p_T.rename("pT"), on=[gcol, ycol], how="left")
        )

        pD = tmp["pD"].fillna(0.0).to_numpy(dtype=float)
        pT = tmp["pT"].fillna(0.0).to_numpy(dtype=float)

        denom = np.maximum(pT, self.reg_eps)
        r = np.divide(pD, denom, out=np.ones_like(pD), where=denom > 0)
        r = np.minimum(r, self.reg_cap)

        alpha_eff = self._reg_strength(T)

        w = np.ones_like(r)
        mask = pD > 0
        w[mask] = 1.0 + alpha_eff * (r[mask] - 1.0)

        w = np.nan_to_num(w, nan=1.0, posinf=self.reg_cap, neginf=1.0)
        return w

    # ------------------------------------------------------------------
    # BO feature matrix
    # ------------------------------------------------------------------
    def _build_feature_matrix(self, U: pd.DataFrame, T: pd.DataFrame) -> np.ndarray:
        feats = []

        # disagreement feature
        if self.model_low is not None and self.model_up is not None:
            d = np.abs(self.model_up(U) - self.model_low(U))
        else:
            d = np.zeros(len(U), dtype=float)
        feats.append(d.reshape(-1, 1))

        # optional gradient feature
        if self.gradient_fn is not None:
            g = np.nan_to_num(self.gradient_fn(U, T), nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(np.asarray(g, dtype=float).reshape(-1, 1))

        # surrogate features (already matrix-shaped)
        if self.surrogate_feat_fn is not None:
            sf = np.nan_to_num(self.surrogate_feat_fn(U, T), nan=0.0, posinf=0.0, neginf=0.0)
            sf = np.asarray(sf, dtype=float)
            if sf.ndim == 1:
                sf = sf.reshape(-1, 1)
            feats.append(sf)

        X = np.concatenate(feats, axis=1) if len(feats) else np.zeros((len(U), 1), dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(float)

    # ------------------------------------------------------------------
    # GP acquisition (refit whenever BO dataset changed)
    # ------------------------------------------------------------------
    def _gp_ucb(self, X_cand: np.ndarray) -> np.ndarray:
        X = self.bo_state.get("X", None)
        y = self.bo_state.get("y", None)

        # fallback: simple sum of features (keeps "disagreement-like" behavior early)
        if X is None or y is None or len(X) < self.bo_min_points:
            acq0 = np.sum(np.asarray(X_cand, dtype=float), axis=1)
            # tiny noise to avoid deterministic ties
            acq0 = acq0 + 1e-9 * self.rng.randn(len(acq0))
            return acq0.astype(float)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        # refit GP if dataset size changed (fixes stale GP)
        need_refit = True
        gp = self.bo_state.get("gp", None)
        gp_n = self.bo_state.get("gp_n", None)
        if gp is not None and gp_n is not None and int(gp_n) == int(len(y)):
            need_refit = False

        if need_refit:
            gp = GaussianProcessRegressor(
                kernel=C(1.0) * RBF(1.0),
                alpha=1e-6,
                normalize_y=True,
                random_state=self.rng,
            )
            gp.fit(X, y)
            self.bo_state["gp"] = gp
            self.bo_state["gp_n"] = int(len(y))

        mu, sigma = gp.predict(np.asarray(X_cand, dtype=float), return_std=True)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        if self.bo_acq == "ucb":
            return (mu + self.bo_beta * sigma).astype(float)

        # default to UCB if unknown
        return (mu + self.bo_beta * sigma).astype(float)

    # ------------------------------------------------------------------
    # Diversity-aware top-k (MMR-style penalty in feature space)
    # ------------------------------------------------------------------
    def _select_topk_diverse(
        self,
        score: np.ndarray,
        X: Optional[np.ndarray],
        k: int,
        gamma: float,
    ) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        n = len(score)
        if k <= 0 or n == 0:
            return np.array([], dtype=int)
        if k >= n:
            return np.arange(n, dtype=int)

        if X is None or gamma <= 0.0:
            return np.argsort(-score)[:k]

        X = np.asarray(X, dtype=float)
        # normalize for cosine similarity
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + self.eps)

        # only consider a shortlist for speed/stability
        m = int(min(n, max(10 * k, 128)))
        cand = np.argsort(-score)[:m]

        chosen = []
        chosen_X = []

        # greedy MMR
        for _ in range(k):
            best_idx = None
            best_val = -np.inf

            for j in cand:
                if j in chosen:
                    continue
                val = float(score[j])

                if chosen_X:
                    v = Xn[j]
                    sims = [float(np.dot(v, u)) for u in chosen_X]
                    max_sim = max(sims) if sims else 0.0
                    val = val - gamma * max_sim

                if val > best_val:
                    best_val = val
                    best_idx = int(j)

            if best_idx is None:
                break

            chosen.append(best_idx)
            chosen_X.append(Xn[best_idx])

        # fill if greedy stops early
        if len(chosen) < k:
            rest = [int(j) for j in cand if int(j) not in chosen]
            need = k - len(chosen)
            chosen.extend(rest[:need])

        return np.asarray(chosen, dtype=int)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def select_next_batch(self, D: pd.DataFrame, T: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0:
            return D.iloc[0:0]

        queried = set(T["id"]) if len(T) > 0 else set()
        U = D[~D["id"].isin(queried)].copy()
        if len(U) == 0:
            return U

        k = min(k, len(U))

        # --------------------------------------------------------------
        # RANDOM
        # --------------------------------------------------------------
        if self.strategy == "random":
            return U.sample(k, random_state=self.rng).reset_index(drop=True)

        # --------------------------------------------------------------
        # STRATIFIED
        # --------------------------------------------------------------
        if self.strategy == "stratified":
            return sample_stratified_fixed_size(
                df=D,
                n=len(T) + k,
                rng=self.rng,
                group_col=self.group_col,
                label_col=self.label_col,
            ).query("id not in @queried").head(k).reset_index(drop=True)

        # shared: distribution weights
        w = self._distribution_weights(D, T, U)
        self.bo_state["last_rho"] = float(np.mean(w)) if len(w) else 1.0

        # --------------------------------------------------------------
        # DISAGREEMENT
        # --------------------------------------------------------------
        if self.strategy == "disagreement":
            if self.model_low is None or self.model_up is None:
                raise ValueError("disagreement strategy requires model_low and model_up")

            info = np.abs(self.model_up(U) - self.model_low(U))
            info = np.nan_to_num(info, nan=0.0, posinf=0.0, neginf=0.0)

            score = info * w

            self.last_batch_info_raw_ = info.copy()
            self.last_batch_acq_raw_ = None
            self.last_batch_combined_scores_ = score.copy()
            self.last_batch_features_ = None
            self.last_batch_ids_ = None

            idx = np.argsort(-score)[:k]
            return U.iloc[idx].reset_index(drop=True)

        # --------------------------------------------------------------
        # EXPECTED WIDTH
        # --------------------------------------------------------------
        if self.strategy == "expected_width_reduction":
            if self.expected_width_fn is None:
                raise ValueError("expected_width_reduction requires expected_width_fn")

            info = np.nan_to_num(self.expected_width_fn(U, T), nan=0.0, posinf=0.0, neginf=0.0)
            score = info * w

            self.last_batch_info_raw_ = info.copy()
            self.last_batch_acq_raw_ = None
            self.last_batch_combined_scores_ = score.copy()
            self.last_batch_features_ = None
            self.last_batch_ids_ = None

            idx = np.argsort(-score)[:k]
            return U.iloc[idx].reset_index(drop=True)

        # --------------------------------------------------------------
        # BO / BO-HYBRID (disagreement-anchored, BO as stabilizer)
        # --------------------------------------------------------------
        if self.strategy in {"bo", "bo_hybrid"}:
            if self.model_low is None or self.model_up is None:
                raise ValueError("bo strategies require model_low and model_up")

            # anchor signal: disagreement (robust)
            dis = np.abs(self.model_up(U) - self.model_low(U))
            dis = np.nan_to_num(dis, nan=0.0, posinf=0.0, neginf=0.0)

            # auxiliary signal (optional): expected width for bo_hybrid
            if self.strategy == "bo_hybrid":
                if self.expected_width_fn is None:
                    # fall back gracefully to disagreement-only
                    aux = dis.copy()
                else:
                    aux = np.nan_to_num(self.expected_width_fn(U, T), nan=0.0, posinf=0.0, neginf=0.0)
                base_info = 0.5 * dis + 0.5 * aux
            else:
                base_info = dis

            # optional restriction: only run BO inside high-disagreement region
            if self.bo_restrict_quantile > 0.0 and len(dis) > 10:
                q = float(np.quantile(dis, self.bo_restrict_quantile))
                mask_keep = dis >= q
            else:
                mask_keep = np.ones(len(U), dtype=bool)

            # BO acquisition on features
            X = self._build_feature_matrix(U, T)
            acq = self._gp_ucb(X)
            acq = np.nan_to_num(acq, nan=0.0, posinf=0.0, neginf=0.0)

            # normalize BO so it doesn't collapse / dominate numerically
            acq01 = self._sigmoid01(self._zscore(acq))

            # BO mixing schedule (0 early; small later)
            lam = self._bo_mix_lambda(T)

            # Final combined informativeness (anchor + BO stabilizer)
            # - keep base_info dominant
            # - only apply BO in keep-region if restriction is on
            combined = base_info.copy()
            if lam > 0:
                combined = (1.0 - lam) * base_info + lam * acq01
            if self.bo_restrict_quantile > 0.0:
                # outside keep-region: fall back to pure anchor
                combined[~mask_keep] = base_info[~mask_keep]

            # apply distribution regularization
            score = combined * w

            # diversity-aware top-k (in feature space)
            idx = self._select_topk_diverse(
                score=score,
                X=X,
                k=k,
                gamma=self.bo_diversity_gamma,
            )

            # logging buffers over U
            self.last_batch_info_raw_ = base_info.copy()
            self.last_batch_acq_raw_ = combined.copy()           # what you optimize after mixing
            self.last_batch_combined_scores_ = score.copy()
            self.last_batch_features_ = X[idx].copy()
            self.last_batch_ids_ = U.iloc[idx]["id"].to_numpy()

            return U.iloc[idx].reset_index(drop=True)

        # --------------------------------------------------------------
        # fallback (shouldn't happen)
        # --------------------------------------------------------------
        return sample_stratified_fixed_size(
            df=D,
            n=len(T) + k,
            rng=self.rng,
            group_col=self.group_col,
            label_col=self.label_col,
        ).query("id not in @queried").head(k).reset_index(drop=True)
