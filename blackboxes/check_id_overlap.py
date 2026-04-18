import pandas as pd
from blackbox_api_bias_in_bios import BiasInBiosBlackBox, load_bias_in_bios_with_ids

CSV = "blackbox_bios.csv"

bb = BiasInBiosBlackBox(CSV)
hf = load_bias_in_bios_with_ids()

# --- get ids from your BB implementation ---
# most likely: dict id -> row index
bb_ids = set(bb._id_to_row.keys())

# --- get ids from HF ---
hf_ids = set(hf["id"].astype(str).values)

inter = bb_ids & hf_ids
missing_in_bb = hf_ids - bb_ids
extra_in_bb = bb_ids - hf_ids

print("HF:", len(hf_ids))
print("BB:", len(bb_ids))
print("Intersection:", len(inter))
print("HF missing in BB:", len(missing_in_bb))
print("BB not in HF:", len(extra_in_bb))
print("Example missing:", sorted(list(missing_in_bb))[:20])
