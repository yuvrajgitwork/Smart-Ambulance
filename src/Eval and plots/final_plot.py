import pandas as pd
import matplotlib.pyplot as plt

raw = pd.read_csv("data/raw/patient_001_artifacts.csv")
cleaned = pd.read_csv("data/processed/patient_001_cleaned.csv")

t = raw.index

# -------- SpO2 --------
plt.figure()
plt.plot(t, raw["spo2"], label="Raw")
plt.plot(t, cleaned["spo2"], label="Cleaned")
plt.title("SpO2 Before vs After Artifact Cleaning")
plt.xlabel("Time (seconds)")
plt.ylabel("SpO2 (%)")
plt.legend()
plt.show()

# -------- HR --------
plt.figure()
plt.plot(t, raw["hr"], label="Raw")
plt.plot(t, cleaned[""], label="Cleaned")
plt.title("Heart Rate Before vs After Artifact Cleaning")
plt.xlabel("Time (seconds)")
plt.ylabel("HR (bpm)")
plt.legend()
plt.show()

# -------- BP Systolic --------
plt.figure()
plt.plot(t, raw["bp_sys"], label="Raw")
plt.plot(t, cleaned["bp_sys"], label="Cleaned")
plt.title("BP Systolic Before vs After Artifact Cleaning")
plt.xlabel("Time (seconds)")
plt.ylabel("BP Systolic (mmHg)")
plt.legend()
plt.show()

# -------- Motion --------
plt.figure()
plt.plot(t, raw["Motion"], label="Motion")
plt.title("Motion / Vibration Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Motion Level")
plt.legend()
plt.show()