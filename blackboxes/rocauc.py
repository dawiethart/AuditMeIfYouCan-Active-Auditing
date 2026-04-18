#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import roc_auc_score

from blackbox_api_bias_in_bios import (
    BiasInBiosBlackBox, load_bias_in_bios_with_ids, PROF_ID_TO_NAME
)

def main():
    bb = BiasInBiosBlackBox("blackbox_bios.csv", verbose=False)
    hf = load_bias_in_bios_with_ids()

    # filter to BB-covered IDs
    hf = hf[hf["id"].astype(str).apply(bb.has_id)].copy()

    target_prof_id = 21
    target_label = PROF_ID_TO_NAME[target_prof_id]
    j = bb.labels.index(target_label)

    # build minimal D (1000) with numeric id
    hf["id_int"] = hf["id"].str.replace("ID", "", regex=False).astype(int)
    hf["text"] = hf["hard_text"].astype(str)
    hf["group"] = hf["gender"].astype(int)
    hf["true_label"] = (hf["profession"].astype(int) == target_prof_id).astype(int)

    D = hf.sample(n=1000, random_state=0).copy()
    ids_str = [f"ID{i}" for i in D["id_int"].tolist()]
    P = bb.query_distribution(ids_str)
    D["bb_score"] = P[:, j]

    # group-wise AUC + delta
    for g in [0, 1]:
        sub = D[D["group"] == g]
        if sub["true_label"].nunique() < 2:
            print("group", g, "AUC undefined (only one class present)")
        else:
            auc = roc_auc_score(sub["true_label"], sub["bb_score"])
            print("AUC group", g, "=", auc)

if __name__ == "__main__":
    main()
