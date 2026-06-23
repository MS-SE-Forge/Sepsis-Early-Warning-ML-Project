"""
Section: Methodology - Early-Warning Eligibility Tagging

Following the diagnostic in onset_window_diagnostic.py, we found that
~25% of septic patients have onset within the first WINDOW hours of their
ICU stay. This means they cannot logically contribute a genuine pre-onset 
"early warning" training/evaluation example.

Rather than silently blending these patients into one overall evaluation number 
(which would drag down our apparent success rate unfairly), we tag every patient 
into exactly one of three groups:

  - "never_septic"          : SepsisLabel never reaches 1.
  - "early_warning_eligible": Septic, with >= WINDOW hours of pre-onset data.
  - "immediate_only"        : Septic, with < WINDOW hours of pre-onset data
                              (onset happens too early in the stay for a
                              true early-warning example to exist).

This tag is computed ONCE per patient and merged into the windowed feature
dataframe, allowing us to report metrics overall AND stratified by group.
"""
import pandas as pd
import numpy as np

WINDOW = 6  # must match the window used in features_split.py

def tag_patient_eligibility(df, window=WINDOW):
    """
    Evaluates the raw timeline of each patient and assigns them to one of the 
    three structural eligibility groups.
    
    Args:
        df (pd.DataFrame): the RAW (not yet windowed) dataframe, one row per patient-hour,
                           with columns patient_id and SepsisLabel.
        window (int): The lookback window size.
        
    Returns:
        pd.DataFrame: A mapping of patient_id -> eligibility_group.
    """
    records = []
    
    for pid, pdf in df.groupby("patient_id"):
        # Ensure chronological order
        pdf = pdf.sort_values("ICULOS")
        labels = pdf["SepsisLabel"].values
        
        # 1. Did they ever get sepsis?
        if labels.max() == 0:
            group = "never_septic"
        else:
            # 2. If they did get sepsis, how soon did it happen?
            onset_idx = labels.argmax()  # row position (0-indexed) of first label=1
            
            if onset_idx >= window:
                group = "early_warning_eligible" 
            else:
                group = "immediate_only"
                
        records.append({"patient_id": pid, "eligibility_group": group})
        
    return pd.DataFrame(records)

def attach_eligibility(feature_df, eligibility_df):
    """
    Merges the static patient-level eligibility tag onto the row-level 
    windowed feature dataframe.
    
    Args:
        feature_df (pd.DataFrame): The engineered features.
        eligibility_df (pd.DataFrame): The output of `tag_patient_eligibility`.
        
    Returns:
        pd.DataFrame: The engineered features with the new tag column.
    """
    # Left merge ensures we don't lose any feature rows
    merged = feature_df.merge(eligibility_df, on="patient_id", how="left")
    
    # Sanity check: ensure every row successfully received a tag
    assert merged["eligibility_group"].isna().sum() == 0, \
        "Some patients in feature_df have no eligibility tag - check patient_id matching"
        
    return merged

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("eda")
    feat_mod = import_module("features_split")
    fast_mod = import_module("features_fast")

    # Load raw data and tag it immediately
    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = tag_patient_eligibility(df)

    print("Patient counts by eligibility group:")
    print(elig_df["eligibility_group"].value_counts())
    print(f"\nTotal patients: {len(elig_df)}")

    # Build windowed features using the FAST vectorized version
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    
    # Attach the tags to the newly built features
    feat_df = attach_eligibility(feat_df, elig_df)

    print("\nRow counts by eligibility group (after windowing):")
    print(feat_df["eligibility_group"].value_counts())
    
    print("\nPositive rate (label=1) within each group's rows:")
    print(feat_df.groupby("eligibility_group")["label"].mean())
