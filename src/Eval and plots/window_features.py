import numpy as np
import pandas as pd

WINDOW = 30   # 30-second windows
STEP   = 5    # slide every 5 seconds (overlap)

def make_windows(df):
    rows = []

    N = len(df)

    for start in range(0, N - WINDOW, STEP):
        seg = df.iloc[start:start+WINDOW]

        row = {}

        # =========================
        # LEVEL FEATURES
        # =========================
        row["hr_mean"]   = seg["hr"].mean()
        row["spo2_mean"] = seg["spo2"].mean()
        row["sbp_mean"]  = seg["sbp"].mean()
        row["dbp_mean"]  = seg["dbp"].mean()

        # =========================
        # VARIABILITY FEATURES
        # =========================
        row["hr_std"]   = seg["hr"].std()
        row["spo2_std"] = seg["spo2"].std()
        row["sbp_std"]  = seg["sbp"].std()
        row["dbp_std"]  = seg["dbp"].std()

        # =========================
        # TREND FEATURES (slope)
        # =========================
        t = np.arange(len(seg))

        def slope(y):
            if np.std(y) < 1e-3:
                return 0.0
            return np.polyfit(t, y, 1)[0]

        row["hr_slope"]   = slope(seg["hr"].values)
        row["spo2_slope"] = slope(seg["spo2"].values)
        row["sbp_slope"]  = slope(seg["sbp"].values)
        row["dbp_slope"]  = slope(seg["dbp"].values)

        # =========================
        # RANGE / EXTREMES
        # =========================
        row["hr_min"]   = seg["hr"].min()
        row["hr_max"]   = seg["hr"].max()

        row["spo2_min"] = seg["spo2"].min()
        row["spo2_max"] = seg["spo2"].max()

        # =========================
        # COUPLING (PHYSIO LOGIC)
        # =========================
        # HR-SpO2 coupling (stress/desat patterns)
        row["hr_spo2_corr"] = seg["hr"].corr(seg["spo2"])

        # BP pulse pressure variability proxy
        pulse_pressure = seg["sbp"] - seg["dbp"]
        row["pp_mean"] = pulse_pressure.mean()
        row["pp_std"]  = pulse_pressure.std()

        # =========================
        # MOTION CONTEXT
        # =========================
        row["motion_mean"] = seg["motion"].mean()
        row["motion_max"]  = seg["motion"].max()

        rows.append(row)

    feats = pd.DataFrame(rows)

    # Fill NaNs from correlations/std edge cases
    feats = feats.fillna(0)

    return feats

