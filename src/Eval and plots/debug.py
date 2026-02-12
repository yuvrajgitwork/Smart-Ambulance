import pandas as pd
import numpy as np

CLEAN_PATH = "data/raw/patient_001_clean.csv"
ARTI_PATH  = "data/raw/patient_001_artifacts.csv"

print("=== LOADING DATA ===")
clean = pd.read_csv(CLEAN_PATH)
arti  = pd.read_csv(ARTI_PATH)

# =========================
# 1. RATE OF CHANGE SANITY (CLEAN)
# =========================
for df, name in [(clean, "CLEAN"), (arti, "ARTIFACT")]:
    df["spo2_delta"] = df["spo2"].diff().abs()
    df["hr_delta"]   = df["hr"].diff().abs()
    df["sbp_delta"]  = df["sbp"].diff().abs()
    df["dbp_delta"]  = df["dbp"].diff().abs()

    print(f"\n=== MAX DELTAS ({name}) ===")
    print("SpO2 max delta:", df["spo2_delta"].max())
    print("HR   max delta:", df["hr_delta"].max())
    print("SBP  max delta:", df["sbp_delta"].max())
    print("DBP  max delta:", df["dbp_delta"].max())

# =========================
# 2. HOW OFTEN CLEAN DATA LOOKS 'ARTIFACT-LIKE'
# =========================
MOTION_THRESH = 0.12

clean["artifact_like_spo2"] = (
    (clean["motion"] > MOTION_THRESH) &
    (clean["spo2"].diff().abs() > 2)
)

clean["artifact_like_hr"] = (
    (clean["motion"] > MOTION_THRESH) &
    (clean["hr"].diff().abs() > 15)
)

print("\n=== CLEAN DATA ARTIFACT-LIKE COUNTS ===")
print("SpO2 artifact-like points (clean):", clean["artifact_like_spo2"].sum())
print("HR   artifact-like points (clean):", clean["artifact_like_hr"].sum())

# =========================
# 3. SEPARATION: CLEAN vs ARTIFACT DISTRIBUTIONS
# =========================
print("\n=== DELTA DISTRIBUTION COMPARISON (95th percentile) ===")

for col in ["spo2_delta", "hr_delta", "sbp_delta", "dbp_delta"]:
    print(f"{col}:")
    print("  CLEAN 95%:", clean[col].quantile(0.95))
    print("  ARTI  95%:", arti[col].quantile(0.95))

# =========================
# 4. PHASE TRANSITION CHECK
# =========================
print("\n=== PHASE TRANSITION JUMPS (CLEAN) ===")

clean["phase_change"] = clean["clinical_phase"] != clean["clinical_phase"].shift()

transitions = clean[clean["phase_change"]].index.tolist()

for idx in transitions[:5]:
    print(f"\nTransition around index {idx}:")
    window = clean.loc[max(0, idx-3):idx+3, 
                        ["timestamp","clinical_phase","spo2","hr","sbp","dbp"]]
    print(window)

print("\n=== DEBUG COMPLETE ===")
