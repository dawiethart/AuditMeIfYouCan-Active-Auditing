#!/usr/bin/env python3
# ============================================================
# smoketest_bias_in_bios.py
#
# Verifies:
#  1) HF IDs and BB IDs overlap as expected (BB subset)
#  2) After filtering HF to BB IDs, random samples:
#       - bb gender == hf gender
#       - bb gold == hf profession mapped to name (if possible)
#       - prob vectors sane (finite, >=0, sum~1)
#  3) Batch query shape works
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox_api_bias_in_bios import BiasInBiosBlackBox, load_bias_in_bios_with_ids, PROF_ID_TO_NAME


def _hf_prof_to_name(x) -> str:
    # HF may store profession as int id; map via PROF_ID_TO_NAME when possible.
    if pd.isna(x):
        return "nan"
    try:
        xi = int(x)
        return PROF_ID_TO_NAME.get(xi, str(xi))
    except Exception:
        return str(x)


def main():
    # ---- configure paths ----
    CSV_PATH = "blackbox_bios.csv"

    # ---- load blackbox ----
    bb = BiasInBiosBlackBox(CSV_PATH, normalize_from_0_100=True, drop_invalid_rows=True, verbose=True)
    bb_ids = set(bb.ids)
    print("BB rows:", len(bb.ids))
    print("BB labels:", len(bb.labels))

    # ---- load HF + recreate IDs ----
    hf = load_bias_in_bios_with_ids()
    hf["id"] = hf["id"].astype(str)
    print("HF rows:", len(hf))
    print("HF columns:", list(hf.columns))

    # ---- overlap check ----
    hf_ids = set(hf["id"].values)
    inter = hf_ids & bb_ids
    print("Intersection:", len(inter))
    print("HF missing in BB:", len(hf_ids - bb_ids))
    print("BB not in HF:", len(bb_ids - hf_ids))
    if len(hf_ids - bb_ids) > 0:
        ex = list(sorted(hf_ids - bb_ids))[:20]
        print("Example missing:", ex)

    # ---- filter HF to BB IDs (critical step) ----
    hf = hf[hf["id"].isin(bb_ids)].reset_index(drop=True)
    print("Filtered HF to BB:", len(hf))

    # ---- build quick lookup maps ----
    hf_gender = dict(zip(hf["id"].values, pd.to_numeric(hf["gender"], errors="coerce").fillna(-1).astype(int).values))
    hf_prof_name = dict(zip(hf["id"].values, hf["profession"].apply(_hf_prof_to_name).values))

    # ---- random sample from BB ids ----
    rng = np.random.default_rng(0)
    ids = bb.ids
    sample = [ids[i] for i in rng.choice(len(ids), size=200, replace=False)]

    # ---- checks ----
    ok = 0
    for sid in sample:
        # gender
        g_bb = bb.get_gender(sid)
        g_hf = int(hf_gender[sid])
        if g_bb is None:
            raise RuntimeError(f"BB gender missing for {sid}")
        if g_hf == -1:
            raise RuntimeError(f"HF gender missing for {sid}")
        if int(g_bb) != g_hf:
            raise AssertionError(f"Gender mismatch {sid}: bb={g_bb} hf={g_hf}")

        # gold vs profession-name (if BB has gold string)
        gold = bb.get_gold_label(sid)
        if gold is not None:
            prof_name = str(hf_prof_name[sid])
            if gold != prof_name:
                raise AssertionError(f"Gold mismatch {sid}: bb_gold='{gold}' hf_prof='{prof_name}'")

        # distribution sanity
        p = bb.query_distribution(sid)
        if p.ndim != 1:
            raise AssertionError(f"Expected 1D probs, got shape {p.shape} for {sid}")
        if not np.isfinite(p).all():
            raise AssertionError(f"Non-finite probs for {sid}")
        if (p < -1e-7).any():
            raise AssertionError(f"Negative probs for {sid}: min={p.min()}")
        s = float(p.sum())
        if abs(s - 1.0) > 1e-3:
            raise AssertionError(f"Probs do not sum to 1 for {sid}: sum={s}")

        ok += 1

    print(f"Smoketest passed: {ok}/{len(sample)} single-id checks")

    # ---- batch query sanity ----
    batch = sample[:32]
    P = bb.query_distribution(batch)
    print("Batch probs shape:", P.shape)
    print("Batch sums (first 5):", P.sum(axis=1)[:5])

    print("ALL GOOD ✅")


if __name__ == "__main__":
    main()
