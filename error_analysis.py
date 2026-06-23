"""
Section: Error Analysis and Limitations

This script extracts concrete patient-level examples from the TEST set 
(the final held-out evaluation set) across four distinct clinical categories:

  1. SUCCESS: A septic patient caught early, with good actionable lead time.
  2. MISS: A septic patient who had plenty of history but was never flagged before onset.
  3. FALSE ALARM: A non-septic patient whose alarms kept ringing.
  4. STRUCTURAL LIMITATION: An `immediate_only` patient, demonstrating exactly
     why this group cannot be "caught early" by mathematical construction.

We print the raw vital-sign trajectories for these patients to provide exact visibility
into what the algorithm processed when it succeeded or failed.
"""
import numpy as np
import pandas as pd

# The optimal utility threshold selected in earlier steps
THRESHOLD = 0.80  

def _find_success_case(df):
    """
    Finds the median lead-time catch among early_warning_eligible patients.
    This represents a SUCCESS: A septic patient caught early, with good actionable lead time.
    """
    success_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "early_warning_eligible"].groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        if pdf["label"].max() == 0:
            continue
            
        onset_idx = pdf["label"].values.argmax()
        pre_onset_flags = pdf["pred_label"].values[:onset_idx]
        
        if pre_onset_flags.sum() > 0:
            first_flag_idx = np.argmax(pre_onset_flags == 1)
            lead = onset_idx - first_flag_idx
            success_candidates.append((pid, lead))
            
    if success_candidates:
        # We pick a representative (median) case rather than cherry-picking the absolute best
        success_candidates.sort(key=lambda x: x[1])
        return success_candidates[len(success_candidates) // 2]
    return None

def _find_miss_case(df):
    """
    Finds the most "should have been catchable" miss.
    This represents a MISS: A patient with >12 hours of pre-onset history who was still never flagged.
    """
    miss_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "early_warning_eligible"].groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        if pdf["label"].max() == 0:
            continue
            
        onset_idx = pdf["label"].values.argmax()
        pre_onset_flags = pdf["pred_label"].values[:onset_idx]
        
        # Condition: 0 flags, but had at least 12 hours of pre-onset data
        if pre_onset_flags.sum() == 0 and onset_idx >= 12:  
            miss_candidates.append((pid, onset_idx))
            
    if miss_candidates:
        # Pick the one with the most pre-onset history (the biggest failure)
        miss_candidates.sort(key=lambda x: -x[1])  
        return miss_candidates[0]
    return None

def _find_false_alarm_case(df):
    """
    Finds a non-septic patient with the most false-positive hours.
    This represents a FALSE ALARM case.
    """
    fa_candidates = []
    for pid, pdf in df[df["eligibility_group"] == "never_septic"].groupby("patient_id"):
        n_false_alarms = pdf["pred_label"].sum()
        if n_false_alarms > 0:
            fa_candidates.append((pid, n_false_alarms))
            
    if fa_candidates:
        # Sort descending by number of false alarms
        fa_candidates.sort(key=lambda x: -x[1])
        return fa_candidates[0]
    return None

def _find_structural_limit_case(df):
    """
    Finds any immediate_only patient.
    This represents a STRUCTURAL LIMITATION: onset happens too early for early warning.
    """
    imm_patients = df[df["eligibility_group"] == "immediate_only"]["patient_id"].unique()
    if len(imm_patients) > 0:
        return (imm_patients[0], None)
    return None

def find_examples(test_df_with_meta, raw_df, pred_probs, threshold=THRESHOLD,
                   vitals_to_show=("HR", "Resp", "MAP", "SBP")):
    """
    Scans the test set to find representative clinical case studies.
    
    Args:
        test_df_with_meta (pd.DataFrame): Windowed test rows with patient metadata.
        raw_df (pd.DataFrame): The ORIGINAL (pre-windowing) hourly data. We need this 
                               to show the full timeline, including the first 6 "warm-up" hours.
        pred_probs (np.ndarray): Model's predicted probabilities.
        threshold (float): Decision threshold.
        vitals_to_show (tuple): Which columns to extract for the report.
        
    Returns:
        dict: A dictionary containing the patient IDs for the 4 specific case studies.
    """
    df = test_df_with_meta.copy()
    df["pred_prob"] = pred_probs
    df["pred_label"] = (df["pred_prob"] >= threshold).astype(int)

    examples = {}

    success = _find_success_case(df)
    if success:
        examples["success"] = success

    miss = _find_miss_case(df)
    if miss:
        examples["miss"] = miss

    fa = _find_false_alarm_case(df)
    if fa:
        examples["false_alarm"] = fa

    sl = _find_structural_limit_case(df)
    if sl:
        examples["structural_limit"] = sl

    return examples

def print_patient_trajectory(pid, raw_df, vitals=("HR", "Resp", "MAP", "SBP"), label_note=""):
    """
    Prints the raw, hour-by-hour vital sign trajectory for a specific patient.
    
    Args:
        pid (str): The patient ID.
        raw_df (pd.DataFrame): The unwindowed dataset.
        vitals (tuple): The columns to display.
        label_note (str): A descriptive note explaining why this patient was chosen.
    """
    pdf = raw_df[raw_df["patient_id"] == pid].sort_values("ICULOS")
    onset_idx = pdf["SepsisLabel"].values.argmax() if pdf["SepsisLabel"].max() == 1 else None
    
    print(f"\nPatient {pid} {label_note}")
    print(f"ICU stay length: {len(pdf)} hours" +
          (f", sepsis onset at hour {onset_idx + 1}" if onset_idx is not None else " (never septic)"))
          
    cols = ["ICULOS"] + list(vitals) + ["SepsisLabel"]
    print(pdf[cols].to_string(index=False))

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("eda")
    feat_mod = import_module("features_split")
    fast_mod = import_module("features_fast")
    elig_mod = import_module("eligibility_tagging")
    xgb_mod = import_module("xgboost_main")

    # Pipeline execution
    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = elig_mod.tag_patient_eligibility(df)

    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    feat_df = elig_mod.attach_eligibility(feat_df, elig_df)

    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source",
                                  "label", "eligibility_group")]
                                  
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    # Extract the specific case studies
    examples = find_examples(
        test_df[["patient_id", "ICULOS", "label", "eligibility_group"]],
        df,  # Pass raw pre-windowing data to show full trajectories
        test_probs
    )

    # Print the specific patient reports
    if "success" in examples:
        pid, lead = examples["success"]
        print_patient_trajectory(pid, df, label_note=f"[SUCCESS CASE - caught {lead}h before onset]")

    if "miss" in examples:
        pid, onset_idx = examples["miss"]
        print_patient_trajectory(pid, df, label_note=f"[MISS CASE - had {onset_idx}h of pre-onset history, never flagged]")

    if "false_alarm" in examples:
        pid, n_fa = examples["false_alarm"]
        print_patient_trajectory(pid, df, label_note=f"[FALSE ALARM CASE - {n_fa} hours incorrectly flagged]")

    if "structural_limit" in examples:
        pid, _ = examples["structural_limit"]
        print_patient_trajectory(pid, df, label_note="[STRUCTURAL LIMITATION CASE - immediate_only group]")