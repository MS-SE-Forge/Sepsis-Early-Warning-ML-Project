"""
Section: Results - Stratified Evaluation by Eligibility Group

Reports AUROC / AUPRC overall AND separately for:
  - early_warning_eligible patients (the patients our project's "early
    warning" claim genuinely applies to)
  - immediate_only patients (onset too early in stay for true early
    warning; here we're really measuring "fast detection," a different
    and still useful, but DIFFERENT, capability)
  - never_septic patients are folded into both denominators as the
    negative class, same as standard evaluation.

Why we don't just drop immediate_only patients entirely: that would
throw away real patients and real signal, and would make our "headline"
overall numbers look artificially clean. Reporting all three rows is
more honest and more informative.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def stratified_metrics(df_with_preds, prob_col, label_col="label",
                        group_col="eligibility_group"):
    """
    df_with_preds must contain: label_col, prob_col, group_col
    Returns a small results table: one row per group + one "Overall" row.
    """
    rows = []

    def _compute(sub_df, name):
        y_true = sub_df[label_col]
        y_prob = sub_df[prob_col]
        if y_true.nunique() < 2:
            # Can't compute AUROC/AUPRC if a slice has only one class present
            return {"group": name, "n_rows": len(sub_df),
                     "n_positive": int(y_true.sum()),
                     "AUROC": np.nan, "AUPRC": np.nan,
                     "note": "only one class present in this slice"}
        return {"group": name, "n_rows": len(sub_df),
                 "n_positive": int(y_true.sum()),
                 "AUROC": roc_auc_score(y_true, y_prob),
                 "AUPRC": average_precision_score(y_true, y_prob),
                 "note": ""}

    # Overall (all rows, all groups combined) - your headline number
    rows.append(_compute(df_with_preds, "Overall"))

    # Early-warning-eligible septic patients + all never_septic as negatives
    ew_mask = df_with_preds[group_col].isin(["early_warning_eligible", "never_septic"])
    rows.append(_compute(df_with_preds[ew_mask], "Early-warning-eligible (+ negatives)"))

    # Immediate-only septic patients + all never_septic as negatives
    imm_mask = df_with_preds[group_col].isin(["immediate_only", "never_septic"])
    rows.append(_compute(df_with_preds[imm_mask], "Immediate-detection-only (+ negatives)"))

    return pd.DataFrame(rows)

def stratified_lead_time(df_with_preds, prob_col, threshold, label_col="label"):
    """
    Lead-time analysis, but reported separately for early_warning_eligible
    vs immediate_only patients. Immediate_only patients will, by
    definition, show lead times close to 0 -- that's the EXPECTED and
    correct result for that group, not an error.
    """
    df = df_with_preds.copy()
    df["pred_label"] = (df[prob_col] >= threshold).astype(int)

    for group in ["early_warning_eligible", "immediate_only"]:
        sub = df[df["eligibility_group"] == group]
        lead_times, caught, missed = [], 0, 0
        for pid, pdf in sub.groupby("patient_id"):
            pdf = pdf.sort_values("ICULOS")
            onset_idx = pdf[label_col].values.argmax()
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
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")
    fast_mod = import_module("02b_features_fast")
    elig_mod = import_module("08_eligibility_tagging")
    xgb_mod = import_module("04_xgboost_main")

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

    print("\n=== Stratified lead-time (validation set, threshold=0.5) ===")
    stratified_lead_time(val_df, "pred_prob", threshold=0.5)