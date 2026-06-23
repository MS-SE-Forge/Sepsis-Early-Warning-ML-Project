"""
Section: Results - Stratified Evaluation by Eligibility Group

This script reports AUROC / AUPRC metrics overall AND separately for:
  - early_warning_eligible patients (the patients our project's "early
    warning" claim genuinely applies to)
  - immediate_only patients (onset too early in stay for true early
    warning; here we're really measuring "fast detection," a different capability)

Why we don't just drop immediate_only patients entirely:
Throwing away real patients who got sepsis would artificially inflate our overall numbers. 
Reporting all three strata is scientifically honest and highly informative.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def stratified_metrics(df_with_preds, prob_col, label_col="label", group_col="eligibility_group"):
    """
    Computes performance metrics broken down by the structural eligibility groups.
    
    Args:
        df_with_preds (pd.DataFrame): Must contain true labels, predictions, and eligibility tags.
        prob_col (str): The column with model probabilities.
        label_col (str): The true binary label.
        group_col (str): The eligibility tag column.
        
    Returns:
        pd.DataFrame: A results table with one row per group + one "Overall" row.
    """
    rows = []

    def _compute(sub_df, name):
        """Helper to calculate metrics on a specific slice of data."""
        y_true = sub_df[label_col]
        y_prob = sub_df[prob_col]
        
        # If a slice has only 1 class (e.g., only negatives), AUROC mathematically cannot be computed.
        if y_true.nunique() < 2:
            return {"group": name, "n_rows": len(sub_df),
                     "n_positive": int(y_true.sum()),
                     "AUROC": np.nan, "AUPRC": np.nan,
                     "note": "only one class present in this slice"}
                     
        return {"group": name, "n_rows": len(sub_df),
                 "n_positive": int(y_true.sum()),
                 "AUROC": roc_auc_score(y_true, y_prob),
                 "AUPRC": average_precision_score(y_true, y_prob),
                 "note": ""}

    # 1. Overall metrics (all rows, all groups combined) - This is the "headline" number
    rows.append(_compute(df_with_preds, "Overall"))

    # 2. Genuine Early-Warning slice
    # We evaluate early-warning eligible patients against ALL never_septic patients
    ew_mask = df_with_preds[group_col].isin(["early_warning_eligible", "never_septic"])
    rows.append(_compute(df_with_preds[ew_mask], "Early-warning-eligible (+ negatives)"))

    # 3. Immediate-detection slice
    # We evaluate immediate-only patients against ALL never_septic patients
    imm_mask = df_with_preds[group_col].isin(["immediate_only", "never_septic"])
    rows.append(_compute(df_with_preds[imm_mask], "Immediate-detection-only (+ negatives)"))

    return pd.DataFrame(rows)

def stratified_lead_time(df_with_preds, prob_col, threshold, label_col="label"):
    """
    Computes lead-time (hours of early warning) separately for early_warning_eligible
    vs immediate_only patients. 
    
    Immediate_only patients will, by mathematical definition, show lead times close to 0.
    That is the EXPECTED and correct result for that group, not a model error.
    
    Args:
        df_with_preds (pd.DataFrame): Dataset with predictions.
        prob_col (str): Column name for probabilities.
        threshold (float): Decision threshold.
        label_col (str): Column name for true labels.
    """
    df = df_with_preds.copy()
    df["pred_label"] = (df[prob_col] >= threshold).astype(int)

    for group in ["early_warning_eligible", "immediate_only"]:
        sub = df[df["eligibility_group"] == group]
        lead_times, caught, missed = [], 0, 0
        
        for pid, pdf in sub.groupby("patient_id"):
            pdf = pdf.sort_values("ICULOS")
            onset_idx = pdf[label_col].values.argmax()
            
            # Extract the model's binary flags prior to clinical onset
            pre_onset_flags = pdf["pred_label"].values[:onset_idx]
            
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_times.append(onset_idx - first_flag_idx)
                caught += 1
            else:
                missed += 1

        print(f"\n--- {group} ---")
        print(f"Caught before onset: {caught}, Missed: {missed}")
        if lead_times:
            print(pd.Series(lead_times).describe())

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
    
    val_df = val_df.copy()
    val_df["pred_prob"] = val_probs

    print("=== Stratified metrics (validation set) ===")
    results = stratified_metrics(val_df, "pred_prob")
    print(results.to_string(index=False))

    # Use the same threshold validated via utility sweep in threshold_leadtime.py
    THRESHOLD = 0.80
    print("\n=== Stratified lead-time (validation set, threshold=0.80) ===")
    stratified_lead_time(val_df, "pred_prob", threshold=THRESHOLD)