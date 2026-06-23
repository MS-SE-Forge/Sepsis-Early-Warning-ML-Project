"""
Section: Methodology - windowing, feature engineering, patient-level split

This script defines the core feature engineering logic for the time-series data.
It handles clinical missingness (forward filling), extracts window-based features 
(like slopes and means over the last 6 hours), and crucially implements a 
patient-level dataset split to guarantee no data leakage between train and test sets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Define clinical feature groups
VITALS = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp"]
LABS = ["BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
        "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
        "Glucose","Lactate","Magnesium","Phosphate","Potassium",
        "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC",
        "Fibrinogen","Platelets"]
DEMOGRAPHICS = ["Age","Gender","HospAdmTime"]

# The sliding window size in hours for feature extraction
WINDOW = 6  

def add_missingness_indicators(df, cols):
    """
    Create binary indicator columns that flag whether a value was missing.
    In clinical settings, the *absence* of a lab test often holds predictive signal 
    (e.g., doctors only order a lactate test if they suspect sepsis).
    
    Args:
        df (pd.DataFrame): The clinical dataset.
        cols (list): List of column names to create indicators for (usually LABS).
        
    Returns:
        pd.DataFrame: DataFrame with new '{col}_missing' binary columns.
    """
    for c in cols:
        # Create a new column set to 1 if the original value is NaN, else 0
        df[f"{c}_missing"] = df[c].isna().astype(int)
    return df

def forward_fill_within_patient(df, cols):
    """
    Handle missing clinical measurements by carrying forward the last known value.
    This simulates reality: until a new lab result arrives, the doctor assumes
    the patient's state is reasonably close to their last measurement.
    
    Args:
        df (pd.DataFrame): The clinical dataset.
        cols (list): Columns to forward-fill (usually VITALS + LABS).
        
    Returns:
        pd.DataFrame: The imputed DataFrame.
    """
    # Forward-fill (ffill) strictly WITHIN each patient's timeline so we don't 
    # accidentally leak patient A's final heart rate into patient B's first hour.
    df[cols] = df.groupby("patient_id")[cols].ffill()
    
    # Anything still missing at the very start of a stay (no prior value yet to ffill from)
    # gets filled with the population median. This is documented and justified,
    # rather than silently imputed.
    for c in cols:
        df[c] = df[c].fillna(df[c].median())
    return df

def build_windowed_features(df, window=WINDOW):
    """
    For each (patient, hour T), build features summarizing the past `window`
    hours: latest value, mean, and trend (slope) for vitals; latest value
    for labs (since labs change slowly and are sparse).
    
    Label = SepsisLabel at hour T (already pre-shifted 6h before clinical
    onset by PhysioNet's own convention - see project write-up).
    
    Args:
        df (pd.DataFrame): The clinical dataset with forward-filled values.
        window (int): The number of hours of history to look back.
        
    Returns:
        pd.DataFrame: A new DataFrame where each row represents a patient at hour T 
                      with window-aggregated features.
    """
    feature_rows = []
    
    # Iterate through each patient's timeline sequentially
    for pid, pdf in df.groupby("patient_id"):
        # Ensure data is strictly ordered by time
        pdf = pdf.sort_values("ICULOS").reset_index(drop=True)
        n = len(pdf)
        
        # We start at `window` because we need at least `window` hours of history 
        # to calculate meaningful rolling statistics.
        for t in range(window, n):  
            # Extract the slice of the timeline for the current window (inclusive of current hour)
            hist = pdf.iloc[t - window : t + 1]  
            
            # Initialize the feature dictionary for this specific hour T
            row = {"patient_id": pid, "ICULOS": pdf.loc[t, "ICULOS"]}

            # Calculate window statistics for highly dynamic Vital Signs
            for v in VITALS:
                vals = hist[v].values
                row[f"{v}_last"] = vals[-1]                   # The most recent measurement
                row[f"{v}_mean"] = np.mean(vals)              # The average over the window
                # Calculate the linear trend (slope) to detect rising/falling vitals
                row[f"{v}_slope"] = np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 1 else 0.0

            # Calculate window statistics for slower-moving Lab results
            for lab in LABS:
                row[f"{lab}_last"] = hist[lab].values[-1]     # The most recent lab value
                row[f"{lab}_missing_rate"] = hist[f"{lab}_missing"].mean() # How frequently it was missing

            # Demographics are static, so we just pull the current value
            for d in DEMOGRAPHICS:
                row[d] = pdf.loc[t, d]

            row["hospital_source"] = pdf.loc[t, "hospital_source"]
            row["label"] = pdf.loc[t, "SepsisLabel"]
            
            feature_rows.append(row)

    return pd.DataFrame(feature_rows)

def patient_level_split(feature_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the data into Train, Validation, and Test sets.
    
    CRITICAL: The split must be done by `patient_id`, not by row. 
    A random row split would leak future data (e.g., patient A's hour 15 in train, 
    patient A's hour 12 in test), ruining the integrity of the evaluation.
    
    Args:
        feature_df (pd.DataFrame): The dataset containing patient features.
        test_size (float): Proportion of patients allocated to the test set.
        val_size (float): Proportion of patients allocated to the validation set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # GroupShuffleSplit ensures that all rows for a given group (patient_id) 
    # end up in the exact same split.
    
    # 1. Split off the Test set
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(feature_df, groups=feature_df["patient_id"]))
    
    train_val_df = feature_df.iloc[train_val_idx]
    test_df = feature_df.iloc[test_idx]

    # 2. Split the remaining data into Train and Validation
    # Adjust val_size to be a proportion of the remaining data
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df["patient_id"]))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    # Sanity check: confirm zero patient overlap across splits.
    # In clinical ML, always verify leakage dynamically, don't just assume the API worked.
    assert set(train_df.patient_id) & set(val_df.patient_id) == set()
    assert set(train_df.patient_id) & set(test_df.patient_id) == set()
    assert set(val_df.patient_id) & set(test_df.patient_id) == set()

    return train_df, val_df, test_df

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    
    # Test the logic on a single dataset folder
    df = eda.load_all_patients(["training_setA"])
    df = add_missingness_indicators(df, LABS)
    df = forward_fill_within_patient(df, VITALS + LABS)
    
    feat_df = build_windowed_features(df)
    print(f"Built {len(feat_df):,} windowed examples from {feat_df.patient_id.nunique()} patients")
    print(f"Positive rate in windowed examples: {feat_df['label'].mean()*100:.2f}%")

    train_df, val_df, test_df = patient_level_split(feat_df)
    print(f"\nTrain: {len(train_df):,} rows, {train_df.patient_id.nunique()} patients")
    print(f"Val:   {len(val_df):,} rows, {val_df.patient_id.nunique()} patients")
    print(f"Test:  {len(test_df):,} rows, {test_df.patient_id.nunique()} patients")
    
    print("\nNo patient overlap across splits: CONFIRMED (assertions passed)")