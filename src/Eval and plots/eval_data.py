import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/processed")
OUT_SUMMARY = Path("data/processed/cleaning_metrics_summary.csv")

BUFFER_SEC = 2

rows = []

print("=== Batch Evaluation Started ===")

for pid in range(1, 51):
    pid_str = f"{pid:03d}"

    raw_path = RAW_DIR / f"patient_{pid_str}_artifacts.csv"
    clean_path = CLEAN_DIR / f"patient_{pid_str}_cleaned.csv"

    if not raw_path.exists() or not clean_path.exists():
        print(f"⚠️ Missing files for patient_{pid_str} — skipping")
        continue

    raw = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path)

    artifact = (raw["artifact_spo2"] == 1).values
    changed = (raw["spo2"] != clean["spo2"]).values

    expanded_artifact = binary_dilation(artifact, iterations=BUFFER_SEC)

    TP = (artifact & changed).sum()
    FN = (artifact & ~changed).sum()
    FP = (~expanded_artifact & changed).sum()
    TN = (~artifact & ~changed).sum()

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    if "clinical_phase" in raw.columns:
        danger_mask = raw["clinical_phase"].isin(["DISTRESS", "ACUTE"]).values
        dangerous_fps = (~expanded_artifact & changed & danger_mask).sum()
    else:
        dangerous_fps = np.nan

    rows.append({
        "patient_id": pid_str,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "dangerous_FPs": dangerous_fps
    })

    print(f"patient_{pid_str} | P={precision:.3f} R={recall:.3f} FP={FP}")

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_SUMMARY, index=False)

print("\n=== Batch Evaluation Complete ===")
print(f"Saved summary to: {OUT_SUMMARY}")

print("\n=== Overall Dataset Metrics ===")
print(summary_df[["precision","recall","dangerous_FPs"]].describe())
