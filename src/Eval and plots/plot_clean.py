import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/patient_001_clean.csv")

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df["hr"])
plt.title("Heart Rate")

plt.subplot(4, 1, 2)
plt.plot(df["spo2"])
plt.title("SpO2")

plt.subplot(4, 1, 3)
plt.plot(df["sbp"], label="SBP")
plt.plot(df["dbp"], label="DBP")
plt.legend()
plt.title("Blood Pressure")

plt.subplot(4, 1, 4)
plt.plot(df["motion"])
plt.title("Motion")

plt.tight_layout()
plt.show()