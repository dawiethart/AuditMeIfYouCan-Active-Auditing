import numpy as np
import pandas as pd
from typing import Optional

from blackboxes.blackbox_api_bias_in_bios import load_bias_in_bios_with_ids, PROF_ID_TO_NAME


PROF_NAME_TO_ID = {v: k for k, v in PROF_ID_TO_NAME.items()}

def load_jigsaw(path="jigsaw_group.csv", groups=("white", "black")):
    df = pd.read_csv(path)
    df["id"] = df["id"].astype("category").cat.codes
    df = df[df["group"].isin(groups)].copy()
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    df["weights"] = np.ones(len(df)) / len(df)
    return df


def load_sbic_and_train_api(api, path="SBIC_group.csv"):
    df = pd.read_csv(path)
    train_df = df.sample(frac=0.9, random_state=42)
    api.train(train_df["text"], train_df["true_label"], train_df["group"])
    return train_df

import pandas as pd

def load_sbic_and_train_api_df(
    api,
    path: str = "SBIC_group.csv",
    flip_probs=None,
    seed: int = 0,
    train_frac: float = 0.9,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 2e-5,
):
    """
    Loads SBIC, samples train_frac for training with a seed, and fine-tunes the API model
    with biased labels via flip_probs.
    """
    df = pd.read_csv(path)

    # Seeded sample => different training set per trial
    train_df = df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)

    flip_probs = flip_probs or {"black": 0.9, "white": 0.1}

    api.train(
        train_df["text"].tolist(),
        train_df["true_label"].to_numpy(),
        train_df["group"].astype(str).tolist(),
        flip_probs=flip_probs,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed
    )
    return api





def load_bios(
    target_profession: str = "professor",
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      id (int), text (str), group (int 0/1), true_label (int 0/1), profession (int)

    true_label is one-vs-rest: 1 iff profession == target_profession.
    """
    df = load_bias_in_bios_with_ids()  # hard_text, profession, gender, id="ID{i}"

    if "hard_text" not in df.columns:
        raise KeyError("Expected column 'hard_text' in HF Bias-in-Bios.")
    if "profession" not in df.columns:
        raise KeyError("Expected column 'profession' in HF Bias-in-Bios.")
    if "gender" not in df.columns:
        raise KeyError("Expected column 'gender' in HF Bias-in-Bios.")
    if "id" not in df.columns:
        raise KeyError("Expected column 'id' in HF Bias-in-Bios.")

    target_profession = str(target_profession)
    if target_profession not in PROF_NAME_TO_ID:
        raise ValueError(
            f"Unknown target_profession='{target_profession}'. "
            f"Choose one of: {sorted(PROF_NAME_TO_ID.keys())}"
        )
    target_id = int(PROF_NAME_TO_ID[target_profession])

    out = pd.DataFrame()
    out["id"] = df["id"].astype(str).str.replace("ID", "", regex=False).astype(int)
    out["text"] = df["hard_text"].astype(str)
    out["group"] = pd.to_numeric(df["gender"], errors="coerce").fillna(-1).astype(int)
    out["profession"] = pd.to_numeric(df["profession"], errors="coerce").fillna(-1).astype(int)
    out["true_label"] = (out["profession"] == target_id).astype(int)

    # Drop invalid gender/profession rows if any
    out = out[(out["group"].isin([0, 1])) & (out["profession"] >= 0)].reset_index(drop=True)

    # Optional subsample for quick debugging
    if max_rows is not None and len(out) > int(max_rows):
        rng = np.random.RandomState(int(seed))
        out = out.sample(n=int(max_rows), replace=False, random_state=rng).reset_index(drop=True)

    return out
