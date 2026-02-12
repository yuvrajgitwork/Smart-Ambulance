import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

df = pd.read_csv("data/raw/patient_001_artifacts.csv")

t = df.index  # seconds

plt.figure(figsize=(12, 6))
plt.plot(t, df["spo2"], label="SpO2")
plt.plot(t, df["motion"] * 20 + 80, label="Motion (scaled)", alpha=0.6)

plt.title("SpO2 with Motion (Artifacts Visible)")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.show()






# Load the data
raw = pd.read_csv("data/raw/patient_001_artifacts.csv")
clean = pd.read_csv("data/processed/patient_001_cleaned.csv")

# 1. Re-calculate the "Dangerous FP" points
artifact = (raw["artifact_spo2"] == 1).values
changed = (raw["spo2"] != clean["spo2"]).values

# Expand artifact zone by 2 seconds (the buffer)
expanded_artifact = binary_dilation(artifact, iterations=2)

# Find points that are: Changed AND NOT an artifact AND NOT in the buffer
# These are the "Dangerous FPs"
dangerous_fp_mask = changed & ~expanded_artifact

# 2. Setup Plot
plt.figure(figsize=(15, 7))
t = np.arange(len(raw))

# Plot Raw vs Cleaned
plt.plot(t, raw["spo2"], label="Raw Signal (with Artifacts)", color="lightgray", alpha=0.8, linewidth=1)
plt.plot(t, clean["spo2"], label="Cleaned Signal", color="blue", linewidth=1.5)

# Highlight True Artifacts (The ones you injected)
plt.scatter(t[artifact], raw.loc[artifact, "spo2"], color="green", s=10, label="Injected Artifacts", zorder=3)

# Highlight the "Dangerous FPs" (The controversial points)
plt.scatter(t[dangerous_fp_mask], clean.loc[dangerous_fp_mask, "spo2"], 
            color="red", s=40, edgecolors='black', label="DANGEROUS FPs (Cleaned Real Data)", zorder=4)

# Focus on the Distress/Acute Phase
if "clinical_phase" in raw.columns:
    # Find start of distress phase to zoom in
    distress_start = raw[raw["clinical_phase"] == "DISTRESS"].index[0]
    plt.xlim(distress_start - 50, len(raw))
    plt.axvline(distress_start, color='orange', linestyle='--', label="Start of Distress Phase")

plt.title("Clinical Safety Audit: Where is the cleaner removing real data?")
plt.xlabel("Time (seconds)")
plt.ylabel("SpO2 (%)")
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

plt.show()