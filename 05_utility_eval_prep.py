"""
Section: Evaluation Criterion - Utility Score
Formats predictions to the format expected by the OFFICIAL
physionetchallenges/evaluation-2019 evaluate_sepsis_score.py script.

That script expects, per patient:
  - A label file: <PatientID>.psv with columns ["SepsisLabel"], one row/hour
  - A prediction file: <PatientID>.psv with columns ["PredictedProbability","PredictedLabel"]
We provide both, matching row-for-row by ICULOS, so the official scorer can
align them correctly.

THRESHOLD: 0.80, chosen via the approximate utility sweep in
06_threshold_leadtime.py (least-negative approximate utility on the
validation set). This is our final, defensible threshold - not an
arbitrary 0.5 default.
"""
import os
import pandas as pd
import numpy as np

def write_label_and_prediction_files(
    test_df_with_meta,   # must include patient_id, ICULOS, label
    pred_probs,          # model's predicted probability, same row order
    threshold,           # decision threshold to convert prob -> 0/1 label
    label_dir,
    pred_dir,
):
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    df = test_df_with_meta.copy()
    df["pred_prob"] = pred_probs
    df["pred_label"] = (pred_probs >= threshold).astype(int)

    for pid, pdf in df.groupby("patient_id"):
        pdf = pdf.sort_values("ICULOS")

        label_path = os.path.join(label_dir, f"{pid}.psv")
        pdf[["label"]].rename(columns={"label": "SepsisLabel"}).to_csv(
            label_path, sep="|", index=False
        )

        pred_path = os.path.join(pred_dir, f"{pid}.psv")
        pdf[["pred_prob", "pred_label"]].rename(
            columns={"pred_prob": "PredictedProbability", "pred_label": "PredictedLabel"}
        ).to_csv(pred_path, sep="|", index=False)

    print(f"Wrote {df.patient_id.nunique()} label files to {label_dir}")
    print(f"Wrote {df.patient_id.nunique()} prediction files to {pred_dir}")
    print(f"\nNext step (run in your terminal):")
    print(f"  python evaluation-2019/evaluate_sepsis_score.py {label_dir} {pred_dir} utility_results.psv")

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
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    # THRESHOLD = 0.80, chosen via threshold sweep in 06_threshold_leadtime.py
    write_label_and_prediction_files(
        test_df_with_meta=test_df[["patient_id", "ICULOS", "label"]],
        pred_probs=test_probs,
        threshold=0.80,
        label_dir="utility_eval/labels",
        pred_dir="utility_eval/predictions",
    )