import sys
sys.path.append(".")

import pandas as pd
import numpy as np
from pathlib import Path
from window_features import make_windows

RAW = Path("data/processed")
OUT = Path("data/processed/window_dataset.csv")

all_rows = []

for f in sorted(RAW.glob("patient_*_cleaned.csv")):
    pid = f.stem.split("_")[1]
    df = pd.read_csv(f)
    
    feats = make_windows(df)
    feats["patient_id"] = pid
    
    all_rows.append(feats)

full = pd.concat(all_rows, ignore_index=True)
full.to_csv(OUT, index=False)

print("Saved:", OUT)
print(full["patient_id"].value_counts())