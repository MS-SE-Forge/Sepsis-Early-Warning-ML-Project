"""
Section: Evaluation Criterion + Results - Threshold sweep & lead-time analysis

This script evaluates different decision thresholds (converting the model's 0-1 probability 
into a hard "sepsis" or "no sepsis" alarm). 

IMPORTANT: `approx_utility` below is a SIMPLIFIED proxy for sweeping thresholds quickly. 
It captures the core clinical idea: 
- Reward early true positives (lead time).
- Penalize false alarms (alarm fatigue).
- Penalize misses (failure to detect).

We use this to find the optimal mathematical threshold (which turns out to be ~0.80)
rather than blindly using the default 0.50, which would generate far too many false alarms.
"""
import numpy as np
import pandas as pd

def approx_utility(df_with_preds, pred_label_col, label_col="label",
                    reward_early=0.05, penalty_false_alarm=-0.02, penalty_miss=-0.5):
    """
    Computes a fast, approximate utility score per patient.
    
    Clinical Logic:
      - If patient develops sepsis AND model flags *before* or at onset -> + reward_early * lead_hours (capped)
      - If patient develops sepsis AND model NEVER flags before onset -> penalty_miss
      - Any flagged hour where the patient is NOT (yet) septic -> penalty_false_alarm per hour
      
    This is intentionally simple to allow fast hyperparameter/threshold sweeps.
    
    Args:
        df_with_preds (pd.DataFrame): Must contain patient_id, ICULOS, true labels, and predictions.
        pred_label_col (str): Column name for the binary prediction.
        label_col (str): Column name for the true label.
        reward_early (float): Positive points awarded per hour of early warning.
        penalty_false_alarm (float): Negative points penalized per false alarm hour.
        penalty_miss (float): Negative points for failing to catch sepsis entirely.
        
    Returns:
        float: The mean per-patient approximate utility score.
    """
    total = 0.0
    n_patients = 0
    
    # Evaluate score strictly on a per-patient timeline
    for pid, pdf in df_with_preds.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        n_patients += 1
        
        # Did this patient actually get sepsis?
        is_septic = pdf[label_col].max() == 1
        
        flagged = pdf[pred_label_col].values
        labels = pdf[label_col].values

        if is_septic:
            # Find the exact hour of clinical onset
            onset_idx = np.argmax(labels == 1)
            
            # Look only at the model's flags *before* clinical onset
            pre_onset_flags = flagged[:onset_idx]
            
            if pre_onset_flags.sum() > 0:
                # The model successfully provided an early warning!
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_hours = onset_idx - first_flag_idx
                
                # Reward the lead time, but cap it at 12 hours (as extreme early warnings 
                # might be clinically unactionable or lucky guesses).
                total += reward_early * min(lead_hours, 12)  
            else:
                # The model missed the sepsis completely
                total += penalty_miss
                
            # False alarms *before* onset cause alarm fatigue. We penalize these.
            # (We subtract 1 so we don't penalize the actual 'catching' flag).
            false_alarms_before = ((flagged[:onset_idx] == 1) & (labels[:onset_idx] == 0)).sum()
            total += penalty_false_alarm * max(false_alarms_before - 1, 0)  
        else:
            # Patient never got sepsis. EVERY flag is a false alarm.
            false_alarms = (flagged == 1).sum()
            total += penalty_false_alarm * false_alarms

    # Return the average score across the entire cohort
    return total / n_patients  

def sweep_thresholds(test_df_meta, pred_probs, thresholds=None):
    """
    Tests a range of probability thresholds (e.g., 0.05, 0.10 ... 0.90) to find 
    which one yields the highest approximate utility score.
    
    Args:
        test_df_meta (pd.DataFrame): Dataset with patient timelines.
        pred_probs (np.ndarray): The raw probabilities from the model.
        thresholds (array-like, optional): Custom thresholds to test.
        
    Returns:
        pd.DataFrame: A table of thresholds and their resulting utility scores.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
        
    results = []
    df = test_df_meta.copy()
    df["pred_prob"] = pred_probs
    
    for t in thresholds:
        # Convert continuous probability into hard 0/1 decision
        df["pred_label"] = (df["pred_prob"] >= t).astype(int)
        
        # Score the decision
        u = approx_utility(df, "pred_label")
        results.append({"threshold": t, "approx_utility": u})
        
    return pd.DataFrame(results)

def lead_time_report(test_df_meta, pred_probs, threshold):
    """
    Generates a detailed clinical report showing how many hours in advance 
    the model caught the sepsis cases.
    
    Args:
        test_df_meta (pd.DataFrame): Patient timelines.
        pred_probs (np.ndarray): Raw model probabilities.
        threshold (float): The chosen decision boundary.
        
    Returns:
        tuple: (lead_times series, caught count, missed count)
    """
    df = test_df_meta.copy()
    df["pred_label"] = (pred_probs >= threshold).astype(int)

    lead_times = []
    missed = 0
    caught = 0
    
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        
        # Only analyze patients who actually got sepsis
        if pdf["label"].max() == 1:
            onset_idx = np.argmax(pdf["label"].values == 1)
            
            # Look at predictions *before* onset
            pre_onset_flags = pdf["pred_label"].values[:onset_idx]
            
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                
                # Calculate hours of lead time
                lead_times.append(onset_idx - first_flag_idx)
                caught += 1
            else:
                missed += 1

    lead_times = pd.Series(lead_times)
    print(f"Septic patients caught before onset: {caught}")
    print(f"Septic patients MISSED (never flagged pre-onset): {missed}")
    
    if len(lead_times) > 0:
        print("\nLead time (hours before onset), among CAUGHT patients only:")
        print(lead_times.describe())
    else:
        print("\nNo patients caught at this threshold - try a lower threshold.")
        
    return lead_times, caught, missed

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("eda")
    feat_mod = import_module("features_split")
    fast_mod = import_module("features_fast")
    elig_mod = import_module("eligibility_tagging")
    xgb_mod = import_module("xgboost_main")

    # Standard data loading and feature engineering pipeline
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
                                  
    # Train the XGBoost model
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)

    # 1. Overall Threshold Sweep
    sweep_df = sweep_thresholds(val_df[["patient_id", "ICULOS", "label"]], val_probs)
    print("=== Threshold sweep (approx utility, validation set, OVERALL) ===")
    print(sweep_df.to_string(index=False))

    best_threshold = sweep_df.loc[sweep_df["approx_utility"].idxmax(), "threshold"]
    print(f"\nBest threshold by approx utility (overall): {best_threshold:.2f}")

    print("\n=== Lead-time report at best threshold, OVERALL (validation set) ===")
    lead_time_report(val_df[["patient_id", "ICULOS", "label"]], val_probs, best_threshold)

    # 2. Subgroup Analysis: Early-Warning-Eligible Only
    # It's important to test if the optimal threshold changes when we remove patients 
    # who arrived at the ICU already septic (the "immediate_only" group).
    ew_mask = val_df["eligibility_group"].isin(["early_warning_eligible", "never_septic"])
    ew_val_df = val_df[ew_mask].reset_index(drop=True)
    ew_probs = val_probs[ew_mask.values]

    sweep_ew = sweep_thresholds(ew_val_df[["patient_id", "ICULOS", "label"]], ew_probs)
    print("\n=== Threshold sweep (approx utility, validation set, EARLY-WARNING-ELIGIBLE ONLY) ===")
    print(sweep_ew.to_string(index=False))

    best_threshold_ew = sweep_ew.loc[sweep_ew["approx_utility"].idxmax(), "threshold"]
    print(f"\nBest threshold by approx utility (early-warning-eligible only): {best_threshold_ew:.2f}")

    print("\n=== Lead-time report at best threshold, EARLY-WARNING-ELIGIBLE ONLY ===")
    lead_time_report(ew_val_df[["patient_id", "ICULOS", "label"]], ew_probs, best_threshold_ew)