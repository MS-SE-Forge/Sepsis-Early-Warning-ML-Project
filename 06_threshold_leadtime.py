"""
Section: Evaluation Criterion + Results - Threshold sweep & lead-time analysis

IMPORTANT: `approx_utility` below is a SIMPLIFIED proxy for sweeping
thresholds quickly. It captures the core idea (reward early true
positives, penalize false alarms, penalize misses) but is NOT a
substitute for the official evaluate_sepsis_score.py. Report your
FINAL number from the official script; use this only to narrow down
candidate thresholds first.
"""
import numpy as np
import pandas as pd

def approx_utility(df_with_preds, pred_label_col, label_col="label",
                    reward_early=0.05, penalty_false_alarm=-0.02, penalty_miss=-0.5):
    """
    df_with_preds must contain: patient_id, ICULOS, label_col, pred_label_col
    Simplified scoring per patient:
      - if patient ever septic AND model flags before/at onset -> + reward_early * lead_hours (capped)
      - if patient ever septic AND model never flags before onset -> penalty_miss
      - any flagged hour where the patient is NOT (yet) septic -> penalty_false_alarm per hour
    This is intentionally simple - explain its limitations vs the official
    score explicitly in your write-up.
    """
    total = 0.0
    n_patients = 0
    for pid, pdf in df_with_preds.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        n_patients += 1
        is_septic = pdf[label_col].max() == 1
        flagged = pdf[pred_label_col].values
        labels = pdf[label_col].values

        if is_septic:
            onset_idx = np.argmax(labels == 1)
            pre_onset_flags = flagged[:onset_idx]
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_hours = onset_idx - first_flag_idx
                total += reward_early * min(lead_hours, 12)  # cap reward at 12h lead, matching official window
            else:
                total += penalty_miss
            # false alarms AFTER onset don't count the same way; skip for simplicity
            false_alarms_before = ((flagged[:onset_idx] == 1) & (labels[:onset_idx] == 0)).sum()
            total += penalty_false_alarm * max(false_alarms_before - 1, 0)  # don't double-penalize the catching flag
        else:
            false_alarms = (flagged == 1).sum()
            total += penalty_false_alarm * false_alarms

    return total / n_patients  # mean per-patient utility (approximate)

def sweep_thresholds(test_df_meta, pred_probs, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    df = test_df_meta.copy()
    df["pred_prob"] = pred_probs
    for t in thresholds:
        df["pred_label"] = (df["pred_prob"] >= t).astype(int)
        u = approx_utility(df, "pred_label")
        results.append({"threshold": t, "approx_utility": u})
    return pd.DataFrame(results)

def lead_time_report(test_df_meta, pred_probs, threshold):
    df = test_df_meta.copy()
    df["pred_label"] = (pred_probs >= threshold).astype(int)

    lead_times = []
    missed = 0
    caught = 0
    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")
        if pdf["label"].max() == 1:
            onset_idx = np.argmax(pdf["label"].values == 1)
            pre_onset_flags = pdf["pred_label"].values[:onset_idx]
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_times.append(onset_idx - first_flag_idx)
                caught += 1
            else:
                missed += 1

    lead_times = pd.Series(lead_times)
    print(f"Septic patients caught before onset: {caught}")
    print(f"Septic patients MISSED (never flagged pre-onset): {missed}")
    if len(lead_times) > 0:
        print(f"\nLead time (hours before onset), among CAUGHT patients only:")
        print(lead_times.describe())
    else:
        print("\nNo patients caught at this threshold - try a lower threshold.")
    return lead_times, caught, missed

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

    # Overall sweep (same as before - this is your headline threshold choice)
    sweep_df = sweep_thresholds(val_df[["patient_id", "ICULOS", "label"]], val_probs)
    print("=== Threshold sweep (approx utility, validation set, OVERALL) ===")
    print(sweep_df.to_string(index=False))

    best_threshold = sweep_df.loc[sweep_df["approx_utility"].idxmax(), "threshold"]
    print(f"\nBest threshold by approx utility (overall): {best_threshold:.2f}")

    print("\n=== Lead-time report at best threshold, OVERALL (validation set) ===")
    lead_time_report(val_df[["patient_id", "ICULOS", "label"]], val_probs, best_threshold)

    # Sweep restricted to early-warning-eligible patients only (+ never_septic
    # as the negative class) - tells us if the OPTIMAL threshold differs for
    # the group our "early warning" claim is actually about
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