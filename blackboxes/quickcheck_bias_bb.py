#!/usr/bin/env python3
import numpy as np
import pandas as pd

from blackbox_api_bias_in_bios import (
    BiasInBiosBlackBox,
    load_bias_in_bios_with_ids,
    PROF_ID_TO_NAME,
)

def main():
    bb = BiasInBiosBlackBox("blackbox_bios.csv", verbose=True)

    hf = load_bias_in_bios_with_ids()
    # hf columns: hard_text, profession, gender, id  (as you saw)
    print("HF:", len(hf), "BB:", len(bb.ids))

    # keep only IDs that exist in BB (because BB dropped invalid rows)
    hf = hf[hf["id"].astype(str).apply(bb.has_id)].copy()
    print("HF after BB filter:", len(hf))

    # choose a target profession for one-vs-rest
    target_prof_id = 21  # professor
    target_label = PROF_ID_TO_NAME[target_prof_id]
    j = bb.labels.index(target_label)
    print("Target:", target_prof_id, target_label, "col_index:", j)

    # sample a few ids and check probabilities
    ex = hf.sample(n=5, random_state=0)
    ids = ex["id"].astype(str).tolist()

    P = bb.query_distribution(ids)            # (B,C)
    s = P[:, j]                               # score for target label
    print("example ids:", ids)
    print("score min/max:", float(s.min()), float(s.max()))
    print("row sum check:", np.max(np.abs(P.sum(axis=1) - 1.0)))

    assert np.isfinite(s).all()
    assert (s >= 0).all() and (s <= 1).all()

if __name__ == "__main__":
    main()
