import pandas as pd

df = pd.read_csv("data/raw/patient_001_artifacts.csv")

print("\nClinical phase counts:")
print(df["clinical_phase"].value_counts())

print("\nBasic stats:")
print(df[["hr", "spo2", "sbp", "dbp", "motion"]].describe())