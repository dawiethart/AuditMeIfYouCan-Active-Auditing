# perspective_offline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd

Id = Union[int, str]


@dataclass
class OfflinePerspectiveBlackBox:
    """
    Offline Perspective 'black-box' backed by precomputed CSV scores.

    Primary key: id (recommended).
    Optional: text lookup as fallback (disabled by default).
    """
    score_by_id: Dict[str, float]
    strict: bool = True

    # Optional text lookup (off by default; can be enabled by passing build_text_index=True)
    score_by_text: Optional[Dict[str, float]] = None

    @classmethod
    def from_csvs(
        cls,
        csv_paths: List[str],
        *,
        id_col: str = "id",
        score_col: str = "score",
        text_col: str = "text",
        strict: bool = True,
        build_text_index: bool = False,
        on_duplicate_id: str = "first",  # "first" | "mean" | "error_if_diff"
        tol: float = 1e-8,
    ) -> "OfflinePerspectiveBlackBox":
        """
        Load and merge multiple group-specific CSVs into one offline black-box.

        on_duplicate_id:
          - "first": keep first occurrence (fast, default)
          - "mean": average scores for same id (only if duplicates exist)
          - "error_if_diff": error if the same id has conflicting scores (>|tol|)
        """
        if len(csv_paths) == 0:
            raise ValueError("csv_paths is empty")

        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

        # Ensure required columns exist
        if id_col not in df.columns:
            raise ValueError(f"Missing required column '{id_col}' in CSV(s).")
        if score_col not in df.columns:
            raise ValueError(f"Missing required column '{score_col}' in CSV(s).")

        # Normalize
        df = df.dropna(subset=[id_col, score_col]).copy()
        df[id_col] = df[id_col].astype(str)
        df[score_col] = df[score_col].astype(float)

        # Handle duplicates across group files
        if df[id_col].duplicated().any():
            if on_duplicate_id == "first":
                df = df.drop_duplicates(subset=[id_col], keep="first")
            elif on_duplicate_id == "mean":
                df = df.groupby(id_col, as_index=False)[score_col].mean()
            elif on_duplicate_id == "error_if_diff":
                # check max-min per id
                g = df.groupby(id_col)[score_col]
                diffs = (g.max() - g.min())
                bad = diffs[diffs > tol]
                if len(bad) > 0:
                    # show a few ids for debugging
                    example_ids = bad.index.tolist()[:10]
                    raise ValueError(
                        f"Found {len(bad)} ids with conflicting scores (tol={tol}). "
                        f"Examples: {example_ids}"
                    )
                df = df.drop_duplicates(subset=[id_col], keep="first")
            else:
                raise ValueError("on_duplicate_id must be one of: first | mean | error_if_diff")

        score_by_id = dict(zip(df[id_col].tolist(), df[score_col].tolist()))

        score_by_text = None
        if build_text_index and (text_col in df.columns):
            # keep first occurrence of each text
            df_text = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col], keep="first")
            score_by_text = dict(zip(df_text[text_col].astype(str), df_text[score_col].astype(float)))

        return cls(score_by_id=score_by_id, strict=strict, score_by_text=score_by_text)

    def coverage(self) -> int:
        return len(self.score_by_id)

    def restrict_df_to_covered_ids(
        self,
        df: pd.DataFrame,
        *,
        id_col: str = "id",
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """
        Filter a candidate pool so every example has an offline score.
        Strongly recommended when strict=True.
        """
        if id_col not in df.columns:
            raise ValueError(f"DataFrame missing '{id_col}' column")

        mask = df[id_col].astype(str).isin(self.score_by_id.keys())
        out = df[mask].copy()
        return out.reset_index(drop=True) if reset_index else out

    def predict_scores(
        self,
        *,
        ids: Optional[Iterable[Id]] = None,
        texts: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """
        Main inference entry point.

        Prefer ids=... (stable).
        texts=... only works if build_text_index=True at load time.
        """
        if ids is None and texts is None:
            raise ValueError("Provide ids=... (recommended) or texts=...")

        if ids is not None:
            out = []
            for _id in ids:
                key = str(_id)
                if key in self.score_by_id:
                    out.append(self.score_by_id[key])
                else:
                    if self.strict:
                        raise KeyError(f"Offline Perspective missing id={_id}")
                    out.append(np.nan)
            return np.asarray(out, dtype=float)

        # Text fallback
        if self.score_by_text is None:
            raise ValueError(
                "Text lookup not available. Load with build_text_index=True or use ids=..."
            )

        out = []
        for t in texts:
            key = str(t)
            if key in self.score_by_text:
                out.append(self.score_by_text[key])
            else:
                if self.strict:
                    raise KeyError("Offline Perspective missing text. Use ids=... instead.")
                out.append(np.nan)
        return np.asarray(out, dtype=float)
