"""
Diagnostic: how many septic patients actually have >= `window` hours of
pre-onset data, given the onset-hour distribution we just observed
(min=1, 25th percentile=7)?

This script investigates a critical structural flaw in the goal of "early warning":
If a patient develops sepsis in hour 2 of their ICU stay, and our model requires 
a 6-hour sliding window of history, it is mathematically impossible to provide 
an "early warning". 

We must diagnose exactly how many patients fall into this "unusable" category 
to ensure our final evaluations are honest.
"""
from importlib import import_module
import pandas as pd

eda = import_module("01_eda")

# The window size used across the project
WINDOW = 6

# Load the raw, unwindowed data
df = eda.load_all_patients(eda.DATA_DIRS)

# Isolate only the patients who actually develop sepsis
septic_ids = df.loc[df["SepsisLabel"] == 1, "patient_id"].unique()

usable = 0
unusable = 0
onset_hours_unusable = []

for pid in septic_ids:
    # Ensure patient timeline is sequential
    pdf = df[df["patient_id"] == pid].sort_values("ICULOS")
    
    # Find the exact row index where sepsis is first labeled
    onset_idx = pdf["SepsisLabel"].values.argmax()  
    
    # If the onset happens *after* the window size, we have enough history
    # to extract features and make a prediction *before* onset occurs.
    if onset_idx >= WINDOW:
        usable += 1
    else:
        # If onset happens *during* the initial window, early warning is impossible.
        unusable += 1
        onset_hours_unusable.append(onset_idx)

# Print the diagnostic findings
print(f"Septic patients with >= {WINDOW}h pre-onset history (usable for early-warning windowing): {usable}")
print(f"Septic patients with < {WINDOW}h pre-onset history (effectively UNUSABLE for early prediction): {unusable}")

# Calculate the critical "impossible" percentage
print(f"-> {unusable / (usable + unusable) * 100:.1f}% of septic patients can't contribute a true 'early warning' positive example with a {WINDOW}h window")

if onset_hours_unusable:
    print("\nOnset row-position distribution among the 'unusable' group:")
    print(pd.Series(onset_hours_unusable).describe())
