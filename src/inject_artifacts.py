import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
DATA_DIR = Path("data/raw")

MOTION_THRESH = 0.12

NUM_SPO2_EVENTS_BASE = 8
NUM_HR_EVENTS_BASE   = 10
NUM_BP_EVENTS_BASE   = 6
NUM_MISSING_EVENTS_BASE = 6

# =========================
# Loop over all patients
# =========================
for patient_id in range(1, 51):

    INPUT_CSV  = DATA_DIR / f"patient_{patient_id:03d}_clean.csv"
    OUTPUT_CSV = DATA_DIR / f"patient_{patient_id:03d}_artifacts.csv"

    if not INPUT_CSV.exists():
        print(f"⚠️ Skipping missing file: {INPUT_CSV}")
        continue

    # -------------------------
    # Seed handling
    # -------------------------
    if patient_id == 1:
        np.random.seed(42)   # KEEP patient_001 EXACT
    else:
        np.random.seed(2000 + patient_id)

    df = pd.read_csv(INPUT_CSV)

    df["artifact_spo2"] = 0
    df["artifact_hr"] = 0
    df["artifact_bp"] = 0
    df["artifact_missing"] = 0

    n = len(df)

    # Patient-specific artifact intensity
    intensity_scale = np.random.uniform(0.7, 1.4)

    num_spo2_events = int(NUM_SPO2_EVENTS_BASE * intensity_scale)
    num_hr_events   = int(NUM_HR_EVENTS_BASE   * intensity_scale)
    num_bp_events   = int(NUM_BP_EVENTS_BASE   * intensity_scale)
    num_missing_events = int(NUM_MISSING_EVENTS_BASE * intensity_scale)

    # =========================
    # 1. Motion-induced SpO2 drops
    # =========================
    for _ in range(num_spo2_events):
        start = np.random.randint(0, n - 20)
        duration = np.random.randint(5, 20)

        for i in range(start, start + duration):
            if df.loc[i, "motion"] > MOTION_THRESH:
                drop = np.random.uniform(5, 12)
                df.loc[i, "spo2"] = max(78, df.loc[i, "spo2"] - drop)
                df.loc[i, "artifact_spo2"] = 1

    # =========================
    # 2. HR spikes due to bumps
    # =========================
    for _ in range(num_hr_events):
        start = np.random.randint(0, n - 10)
        duration = np.random.randint(3, 8)
        spike = np.random.uniform(20, 50)

        for i in range(start, start + duration):
            if df.loc[i, "motion"] > MOTION_THRESH:
                df.loc[i, "hr"] = df.loc[i, "hr"] + spike
                df.loc[i, "artifact_hr"] = 1

    # =========================
    # 3. BP spikes (cuff/motion)
    # =========================
    for _ in range(num_bp_events):
        start = np.random.randint(0, n - 10)
        duration = np.random.randint(3, 8)

        sbp_spike = np.random.uniform(20, 50)
        dbp_spike = np.random.uniform(10, 30)

        for i in range(start, start + duration):
            if df.loc[i, "motion"] > MOTION_THRESH:
                df.loc[i, "sbp"] = df.loc[i, "sbp"] + sbp_spike
                df.loc[i, "dbp"] = df.loc[i, "dbp"] + dbp_spike
                df.loc[i, "artifact_bp"] = 1

    # =========================
    # 4. Missing data segments
    # =========================
    for _ in range(num_missing_events):
        start = np.random.randint(0, n - 15)
        duration = np.random.randint(5, 15)

        for i in range(start, start + duration):
            if np.random.rand() < 0.5:
                df.loc[i, "spo2"] = np.nan
            else:
                df.loc[i, "hr"] = np.nan
            df.loc[i, "artifact_missing"] = 1

    # =========================
    # Save
    # =========================
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Artifacts injected: {OUTPUT_CSV}")
    print(f"   SpO2: {df['artifact_spo2'].sum()} | "
          f"HR: {df['artifact_hr'].sum()} | "
          f"BP: {df['artifact_bp'].sum()} | "
          f"Missing: {df['artifact_missing'].sum()}")

print("\n🎯 Artifact injection complete for all patients.")
