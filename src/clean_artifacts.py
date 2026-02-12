import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG (same as single)
# =========================
MOTION_THRESH = 0.12

SPO2_DROP_THRESH = 5
SPO2_MAX_SEG = 12

HR_JUMP_THRESH = 25
HR_MAX_SEG = 10

SBP_JUMP_THRESH = 25
DBP_JUMP_THRESH = 20
BP_MAX_SEG = 8


def clean_one_patient(input_path, output_path):
    df = pd.read_csv(input_path)

    spo2_fixed = 0
    hr_fixed = 0
    bp_fixed = 0

    N = len(df)

    # Rolling baselines
    df["spo2_base"] = df["spo2"].rolling(30, min_periods=1).median()
    df["hr_base"]   = df["hr"].rolling(30, min_periods=1).median()
    df["sbp_base"]  = df["sbp"].rolling(30, min_periods=1).median()
    df["dbp_base"]  = df["dbp"].rolling(30, min_periods=1).median()

    i = 0
    while i < N:

        # =========================
        # SpO2 SEGMENT
        # =========================
        if (
            df.loc[i, "motion"] > MOTION_THRESH and
            df.loc[i, "spo2"] < df.loc[i, "spo2_base"] - SPO2_DROP_THRESH
        ):
            start = i
            base = df.loc[i, "spo2_base"]

            while i < N and df.loc[i, "spo2"] < base - 2:
                i += 1

            end = i - 1
            dur = end - start + 1

            if 1 <= dur <= SPO2_MAX_SEG:
                df.loc[start:end, "spo2"] = base
                spo2_fixed += dur

            continue

        # =========================
        # HR SEGMENT
        # =========================
        if (
            df.loc[i, "motion"] > MOTION_THRESH and
            abs(df.loc[i, "hr"] - df.loc[i, "hr_base"]) > HR_JUMP_THRESH
        ):
            start = i
            base = df.loc[i, "hr_base"]

            while i < N and abs(df.loc[i, "hr"] - base) > 12:
                i += 1

            end = i - 1
            dur = end - start + 1

            if 1 <= dur <= HR_MAX_SEG:
                df.loc[start:end, "hr"] = base
                hr_fixed += dur

            continue

        # =========================
        # BP SEGMENT
        # =========================
        if (
            df.loc[i, "motion"] > MOTION_THRESH and (
                abs(df.loc[i, "sbp"] - df.loc[i, "sbp_base"]) > SBP_JUMP_THRESH or
                abs(df.loc[i, "dbp"] - df.loc[i, "dbp_base"]) > DBP_JUMP_THRESH
            )
        ):
            start = i
            sbp_base = df.loc[i, "sbp_base"]
            dbp_base = df.loc[i, "dbp_base"]

            while i < N and (
                abs(df.loc[i, "sbp"] - sbp_base) > 12 or
                abs(df.loc[i, "dbp"] - dbp_base) > 8
            ):
                i += 1

            end = i - 1
            dur = end - start + 1

            if 1 <= dur <= BP_MAX_SEG:
                df.loc[start:end, ["sbp", "dbp"]] = [sbp_base, dbp_base]
                bp_fixed += dur

            continue

        i += 1

    df.drop(columns=["spo2_base","hr_base","sbp_base","dbp_base"], inplace=True)
    df.to_csv(output_path, index=False)

    return spo2_fixed, hr_fixed, bp_fixed


# =========================
# Batch Loop
# =========================
print("=== Batch Cleaning Started ===")

for pid in range(1, 51):
    pid_str = f"{pid:03d}"
    in_path = RAW_DIR / f"patient_{pid_str}_artifacts.csv"
    out_path = OUT_DIR / f"patient_{pid_str}_cleaned.csv"

    if not in_path.exists():
        print(f"⚠️ Missing: {in_path} — skipping")
        continue

    spo2_n, hr_n, bp_n = clean_one_patient(in_path, out_path)

    print(f"✅ patient_{pid_str}: SpO2={spo2_n}, HR={hr_n}, BP={bp_n}")

print("=== Batch Cleaning Complete ===")
