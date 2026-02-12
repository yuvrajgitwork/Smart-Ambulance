import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# -----------------------------
# Global Configuration
# -----------------------------
SAMPLE_RATE_HZ = 1
TOTAL_MINUTES = 35
TOTAL_SECONDS = TOTAL_MINUTES * 60

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Base Phase Template (will be jittered per patient)
# -----------------------------
BASE_PHASES = {
    "NORMAL":   (0, 10 * 60),
    "DISTRESS": (10 * 60, 20 * 60),
    "ACUTE":    (20 * 60, 28 * 60),
    "RECOVERY": (28 * 60, TOTAL_SECONDS),
}

BASE_PHASE_TARGETS = {
    "NORMAL":   {"hr": 80,  "spo2": 98, "sbp": 125, "dbp": 80},
    "DISTRESS": {"hr": 105, "spo2": 94, "sbp": 115, "dbp": 75},
    "ACUTE":    {"hr": 135, "spo2": 88, "sbp": 85,  "dbp": 55},
    "RECOVERY": {"hr": 105, "spo2": 94, "sbp": 110, "dbp": 70},
}

# -----------------------------
# Helper functions
# -----------------------------
def smooth_update(prev, target, alpha=0.01, noise_std=0.5):
    return prev + alpha * (target - prev) + np.random.normal(0, noise_std)

def generate_motion(t, motion_scale):
    base_vibration = motion_scale * (0.1 + 0.05 * np.sin(2 * np.pi * t / 60))
    random_jitter = np.random.normal(0, 0.03 * motion_scale)

    if np.random.rand() < 0.01 * motion_scale:
        bump = np.random.uniform(0.3, 0.7) * motion_scale
    else:
        bump = 0.0

    motion = max(0, base_vibration + random_jitter + bump)
    return motion

def build_patient_phases():
    """Randomize phase boundaries per patient"""
    n_end = np.random.randint(8*60, 12*60)
    d_end = n_end + np.random.randint(8*60, 12*60)
    a_end = d_end + np.random.randint(6*60, 10*60)

    return {
        "NORMAL":   (0, n_end),
        "DISTRESS": (n_end, d_end),
        "ACUTE":    (d_end, a_end),
        "RECOVERY": (a_end, TOTAL_SECONDS),
    }

def get_phase(t, PHASES):
    for phase, (start, end) in PHASES.items():
        if start <= t < end:
            return phase
    return "NORMAL"

# -----------------------------
# Main Multi-Patient Generator
# -----------------------------
for patient_id in range(1, 51):

    # -----------------------------
    # Patient-specific random seed
    # -----------------------------
    if patient_id == 1:
        np.random.seed(42)   # KEEP patient_001 EXACT
    else:
        np.random.seed(1000 + patient_id)

    # -----------------------------
    # Patient physiology modifiers
    # -----------------------------
    hr_offset   = np.random.normal(0, 6)
    spo2_offset = np.random.normal(0, 1.0)
    sbp_offset  = np.random.normal(0, 8)
    dbp_offset  = np.random.normal(0, 5)

    motion_scale = np.random.uniform(0.8, 1.4)

    # Phase structure
    if patient_id == 1:
        PHASES = BASE_PHASES
    else:
        PHASES = build_patient_phases()

    # Phase targets (jittered per patient)
    PHASE_TARGETS = {}
    for phase, vals in BASE_PHASE_TARGETS.items():
        PHASE_TARGETS[phase] = {
            "hr":   vals["hr"]   + hr_offset   + np.random.normal(0, 3),
            "spo2": vals["spo2"] + spo2_offset + np.random.normal(0, 0.5),
            "sbp":  vals["sbp"]  + sbp_offset  + np.random.normal(0, 5),
            "dbp":  vals["dbp"]  + dbp_offset  + np.random.normal(0, 3),
        }

    # -----------------------------
    # Time-series containers
    # -----------------------------
    timestamps = []
    hr_series = []
    spo2_series = []
    sbp_series = []
    dbp_series = []
    motion_series = []
    phase_series = []

    # Initial values (patient-specific)
    hr   = 78 + hr_offset
    spo2 = 98.5 + spo2_offset
    sbp  = 128 + sbp_offset
    dbp  = 82 + dbp_offset

    start_time = datetime.now()

    for t in range(TOTAL_SECONDS):
        phase = get_phase(t, PHASES)
        targets = PHASE_TARGETS[phase]

        # Heart Rate
        hr = smooth_update(
            hr, targets["hr"],
            alpha=0.02 if phase in ["DISTRESS", "ACUTE"] else 0.01,
            noise_std=1.2 + 0.2 * motion_scale
        )
        hr = np.clip(hr, 40, 190)

        # SpO2
        spo2 = smooth_update(
            spo2, targets["spo2"],
            alpha=0.01,
            noise_std=0.25
        )
        spo2 = np.clip(spo2, 78, 100)

        # Blood Pressure
        sbp = smooth_update(
            sbp, targets["sbp"] - 0.05 * (hr - 90),
            alpha=0.015,
            noise_std=1.8
        )
        dbp = smooth_update(
            dbp, targets["dbp"] - 0.03 * (hr - 90),
            alpha=0.015,
            noise_std=1.2
        )

        sbp = np.clip(sbp, 60, 190)
        dbp = np.clip(dbp, 35, 130)

        # Motion
        motion = generate_motion(t, motion_scale)

        # Store
        timestamps.append(start_time + timedelta(seconds=t))
        hr_series.append(round(hr, 1))
        spo2_series.append(round(spo2, 2))
        sbp_series.append(round(sbp, 1))
        dbp_series.append(round(dbp, 1))
        motion_series.append(round(motion, 3))
        phase_series.append(phase)

    # -----------------------------
    # Save
    # -----------------------------
    patient_tag = f"patient_{patient_id:03d}_clean.csv"
    output_path = os.path.join(OUTPUT_DIR, patient_tag)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "hr": hr_series,
        "spo2": spo2_series,
        "sbp": sbp_series,
        "dbp": dbp_series,
        "motion": motion_series,
        "clinical_phase": phase_series
    })

    df.to_csv(output_path, index=False)
    print(f"✅ Generated clean data: {output_path}")

print("\n🎯 Multi-patient synthetic dataset complete (50 patients).")
