"""
Section: Results - Subgroup Analysis by Hospital Source (A vs B)

The dataset combines two distinct hospital systems. Reporting one global metric
risks hiding meaningful performance differences between them. 
Testing model performance across different data sources without retraining is a 
core check for model robustness and distribution-shift resilience.

This script reports metrics separately for hospital A and hospital B,
using the SAME trained model (we do not retrain per hospital).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def hospital_subgroup_metrics(df_with_preds, prob_col, label_col="label", source_col="hospital_source"):
    """
    Computes model performance metrics isolated by hospital source.
    
    Args:
        df_with_preds (pd.DataFrame): Dataset with predictions and source tags.
        prob_col (str): Prediction column name.
        label_col (str): True label column name.
        source_col (str): The column indicating hospital system (e.g., A or B).
        
    Returns:
        pd.DataFrame: A table showing metrics per hospital and overall.
    """
    rows = []

    def _compute(sub_df, name):
        """Helper to calculate metrics on a specific slice of data."""
        y_true = sub_df[label_col]
        y_prob = sub_df[prob_col]
        n_patients = sub_df["patient_id"].nunique()
        
        # Handle edge cases where a slice lacks positive or negative examples
        if y_true.nunique() < 2:
            return {"group": name, "n_rows": len(sub_df), "n_patients": n_patients,
                     "n_positive": int(y_true.sum()),
                     "AUROC": np.nan, "AUPRC": np.nan,
                     "note": "only one class present"}
                     
        return {"group": name, "n_rows": len(sub_df), "n_patients": n_patients,
                 "n_positive": int(y_true.sum()),
                 "AUROC": roc_auc_score(y_true, y_prob),
                 "AUPRC": average_precision_score(y_true, y_prob),
                 "note": ""}

    # Append overall and subset metrics
    rows.append(_compute(df_with_preds, "Overall"))
    rows.append(_compute(df_with_preds[df_with_preds[source_col] == "A"], "Hospital A"))
    rows.append(_compute(df_with_preds[df_with_preds[source_col] == "B"], "Hospital B"))

    return pd.DataFrame(rows)

def hospital_subgroup_lead_time(df_with_preds, prob_col, threshold,
                                  label_col="label", source_col="hospital_source"):
    """
    Computes lead-time (hours of early warning) separately per hospital source.
    This helps us answer: "Is the model systematically faster at warning doctors in Hospital A vs B?"
    
    Args:
        df_with_preds (pd.DataFrame): Dataset with predictions.
        prob_col (str): Prediction column.
        threshold (float): Decision threshold.
        label_col (str): True label column.
        source_col (str): Hospital source column.
    """
    df = df_with_preds.copy()
    df["pred_label"] = (df[prob_col] >= threshold).astype(int)

    for source in ["A", "B"]:
        sub = df[df[source_col] == source]
        lead_times, caught, missed = [], 0, 0
        
        for pid, pdf in sub.groupby("patient_id"):
            pdf = pdf.sort_values("ICULOS")
            
            # Skip patients who never get sepsis, as they can't have a lead time
            if pdf[label_col].max() == 0:
                continue  
                
            onset_idx = pdf[label_col].values.argmax()
            pre_onset_flags = pdf["pred_label"].values[:onset_idx]
            
            if pre_onset_flags.sum() > 0:
                first_flag_idx = np.argmax(pre_onset_flags == 1)
                lead_times.append(onset_idx - first_flag_idx)
                caught += 1
            else:
                missed += 1

        print(f"\n--- Hospital {source} ---")
        print(f"Septic patients caught before onset: {caught}, Missed: {missed}")
        if lead_times:
            print(pd.Series(lead_times).describe())

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("eda")
    feat_mod = import_module("features_split")
    fast_mod = import_module("features_fast")
    elig_mod = import_module("eligibility_tagging")
    xgb_mod = import_module("xgboost_main")

    # Execute data pipeline
    df = eda.load_all_patients(eda.DATA_DIRS)
    elig_df = elig_mod.tag_patient_eligibility(df)

    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = fast_mod.build_windowed_features_fast(
        df, feat_mod.VITALS, feat_mod.LABS, feat_mod.DEMOGRAPHICS
    )
    feat_df = elig_mod.attach_eligibility(feat_df, elig_df)

    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    # Sanity Check: confirm both hospitals are represented in train AND test.
    # If our random split accidentally put almost all of one hospital into test, 
    # the results would be highly skewed.
    print("Hospital source distribution across splits:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name}: {split_df['hospital_source'].value_counts().to_dict()}")

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source",
                                  "label", "eligibility_group")]
                                  
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    
    val_df = val_df.copy()
    val_df["pred_prob"] = val_probs

    print("\n=== Hospital subgroup metrics (validation set) ===")
    results = hospital_subgroup_metrics(val_df, "pred_prob")
    print(results.to_string(index=False))

    # Using threshold=0.80, the same optimal utility threshold chosen earlier
    print("\n=== Hospital subgroup lead-time (validation set, threshold=0.80) ===")
    hospital_subgroup_lead_time(val_df, "pred_prob", threshold=0.80)