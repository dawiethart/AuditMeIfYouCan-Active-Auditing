#!/usr/bin/env python3
# ============================================================
# Bias-in-Bios "black-box" wrapper over a CSV of model scores
#
# - Loads CSV with columns:
#     id, gold_occupation, gender, pred_occupation, <score cols...>
# - Exposes:
#     query_distribution(id | [id,...]) -> probs in [0,1], sum=1
#     get_gold_label(id) -> str | None
#     get_gender(id) -> int | None
#     ids -> list[str]
#
# Also includes HF loader that recreates the ID assignment:
#   train, test, dev concat -> reset_index -> id = "ID{i}"
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from datasets import load_dataset


# (Only needed for smoketests / mapping HF profession-id -> name)
PROF_ID_TO_NAME: Dict[int, str] = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher",
}


@dataclass(frozen=True)
class BiasInBiosMeta:
    n_rows_loaded: int
    n_rows_kept: int
    n_rows_dropped: int
    n_score_cols: int


class BiasInBiosBlackBox:
    """
    Efficient black-box over a scores CSV:
      - stores probs in a contiguous float32 matrix
      - stores id -> row index mapping
    """

    def __init__(
        self,
        scores_csv: str,
        normalize_from_0_100: bool = True,
        drop_invalid_rows: bool = True,
        verbose: bool = True,
    ):
        df = pd.read_csv(scores_csv)
        n0 = len(df)

        # Standardize id column
        if "id" not in df.columns:
            raise KeyError("CSV must contain column 'id'.")
        df["id"] = df["id"].astype(str)

        base_cols = {"id", "gold_occupation", "gender", "pred_occupation"}

        # Score columns = everything else, in the CSV order
        score_cols = [c for c in df.columns if c not in base_cols]
        if not score_cols:
            raise ValueError("No score columns detected. Check CSV header.")

        # Make sure required metadata columns exist (gold_occupation might be missing in some variants)
        if "gold_occupation" not in df.columns:
            df["gold_occupation"] = np.nan
        if "gender" not in df.columns:
            df["gender"] = np.nan

        # Coerce score columns to numeric
        for c in score_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Coerce gender to numeric (may contain NaN -> we'll drop or keep as -1)
        gender_num = pd.to_numeric(df["gender"], errors="coerce")

        if drop_invalid_rows:
            # Drop rows with missing id or missing scores or missing gender
            required = ["id"] + score_cols
            df["_gender_num"] = gender_num

            before = len(df)
            df = df.dropna(subset=required + ["_gender_num"])
            # also drop rows with non-finite values (inf)
            finite_mask = np.isfinite(df[score_cols].to_numpy()).all(axis=1) & np.isfinite(df["_gender_num"].to_numpy())
            df = df.loc[finite_mask].copy()
            after = len(df)

            if verbose:
                dropped = before - after
                print(f"[BiasInBiosBlackBox] loaded={n0} kept={after} dropped={dropped} score_cols={len(score_cols)}")

            gender_num = df["_gender_num"]
            df = df.drop(columns=["_gender_num"])
        else:
            # If we don't drop, we keep NaNs as -1 (but then you must handle them downstream)
            gender_num = gender_num.fillna(-1)

        # Build scores matrix
        scores = df[score_cols].to_numpy(dtype=np.float32)

        if normalize_from_0_100:
            scores = scores / np.float32(100.0)

        # Row-wise renormalize to sum=1 (protect against drift / partial probs)
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = np.float32(1.0)
        probs = scores / row_sums

        # Store
        self._df = df.reset_index(drop=True)
        self._score_cols = score_cols
        self._probs = probs  # (N, C) float32

        # IDs
        self._ids = self._df["id"].astype(str).to_numpy()
        # De-duplicate IDs if necessary (keep first occurrence)
        # (If your pipeline guarantees uniqueness, this is a no-op.)
        seen: Dict[str, int] = {}
        keep_rows: List[int] = []
        for i, sid in enumerate(self._ids):
            if sid not in seen:
                seen[sid] = i
                keep_rows.append(i)

        if len(keep_rows) != len(self._ids):
            if verbose:
                print(f"[BiasInBiosBlackBox] WARNING: duplicate IDs found. Keeping first occurrence for each id.")
            keep_rows = np.asarray(keep_rows, dtype=np.int64)
            self._df = self._df.iloc[keep_rows].reset_index(drop=True)
            self._probs = self._probs[keep_rows]
            self._ids = self._df["id"].astype(str).to_numpy()

        self._id_to_row: Dict[str, int] = {sid: i for i, sid in enumerate(self._ids)}

        # Meta columns (optional)
        self._gold = self._df["gold_occupation"].astype(str).to_numpy() if "gold_occupation" in self._df.columns else None
        self._gender = pd.to_numeric(self._df["gender"], errors="coerce").fillna(-1).astype(np.int8).to_numpy()

        self.meta = BiasInBiosMeta(
            n_rows_loaded=n0,
            n_rows_kept=len(self._df),
            n_rows_dropped=n0 - len(self._df),
            n_score_cols=len(self._score_cols),
        )

    # --------- public API ---------

    @property
    def labels(self) -> List[str]:
        return list(self._score_cols)

    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    def has_id(self, id_: str) -> bool:
        return str(id_) in self._id_to_row

    def _row_index(self, id_: str) -> int:
        id_ = str(id_)
        r = self._id_to_row.get(id_)
        if r is None:
            raise KeyError(f"Unknown ID '{id_}' in black-box.")
        return r

    def query_distribution(self, ids: Union[str, Sequence[str]]) -> np.ndarray:
        """
        - if ids is str -> (C,)
        - else -> (B, C)
        """
        if isinstance(ids, str):
            r = self._row_index(ids)
            return self._probs[r].copy()

        rows = np.fromiter((self._row_index(i) for i in ids), dtype=np.int64, count=len(ids))
        return self._probs[rows].copy()

    def get_score_for_label(self, id_: str, label: str) -> float:
        label = str(label)
        try:
            j = self._score_cols.index(label)
        except ValueError as e:
            raise KeyError(f"Unknown label '{label}'. Known: {self._score_cols}") from e
        r = self._row_index(id_)
        return float(self._probs[r, j])

    def get_max_confidence(self, id_: str) -> float:
        r = self._row_index(id_)
        return float(self._probs[r].max())

    def get_gold_label(self, id_: str) -> Optional[str]:
        r = self._row_index(id_)
        if self._gold is None:
            return None
        g = self._gold[r]
        # When gold_occupation was missing it becomes "nan" string; treat as missing
        if g.lower() == "nan":
            return None
        return str(g)

    def get_gender(self, id_: str) -> Optional[int]:
        r = self._row_index(id_)
        g = int(self._gender[r])
        if g == -1:
            return None
        return g


# ---------------- HF: recreate IDs (train, test, dev) ----------------

def load_bias_in_bios_with_ids() -> pd.DataFrame:
    """
    Loads LabHC/bias_in_bios and recreates a stable ID assignment:
      train + test + dev, then id="ID{i}".
    Returns a DataFrame with at least: hard_text, profession, gender, id
    """
    train = pd.DataFrame(load_dataset("LabHC/bias_in_bios", split="train"))
    test = pd.DataFrame(load_dataset("LabHC/bias_in_bios", split="test"))
    dev = pd.DataFrame(load_dataset("LabHC/bias_in_bios", split="dev"))

    df = pd.concat([train, test, dev], ignore_index=True)
    df["id"] = [f"ID{i}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    # Tiny sanity run
    bb = BiasInBiosBlackBox("blackbox_bios.csv", verbose=True)
    sid = bb.ids[0]
    p = bb.query_distribution(sid)
    print("example id:", sid)
    print("probs shape:", p.shape, "sum:", float(p.sum()), "max:", float(p.max()))
    print("gold:", bb.get_gold_label(sid), "gender:", bb.get_gender(sid))
