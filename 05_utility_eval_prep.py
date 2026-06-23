"""
Section: Evaluation Criterion - Utility Score

This script formats our model's predictions to match the exact format expected by the 
OFFICIAL PhysioNet Challenge `evaluate_sepsis_score.py` script. 

The official script expects, per patient:
  - A label file: <PatientID>.psv with column ["SepsisLabel"]
  - A prediction file: <PatientID>.psv with columns ["PredictedProbability","PredictedLabel"]

This ensures our evaluation is 100% compliant with the official 2019 PhysioNet scoring utility.
"""
import os
import pandas as pd
import numpy as np

def write_label_and_prediction_files(
    test_df_with_meta,
    pred_probs,
    threshold,
    label_dir,
    pred_dir,
):
    """
    Writes out the required `.psv` files for the official PhysioNet evaluation script.
    
    Args:
        test_df_with_meta (pd.DataFrame): Subset of the test set containing patient_id, ICULOS, label.
        pred_probs (np.ndarray): Model's raw predicted probability [0.0 to 1.0].
        threshold (float): The chosen decision boundary (e.g. 0.80) to convert probabilities 
                           into hard 0/1 predictions.
        label_dir (str): Directory to save true label files.
        pred_dir (str): Directory to save prediction files.
    """
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # Attach predictions to the metadata frame
    df = test_df_with_meta.copy()
    df["pred_prob"] = pred_probs
    
    # Convert continuous probabilities into a hard clinical decision (0 or 1)
    # using our pre-calculated optimal utility threshold.
    df["pred_label"] = (pred_probs >= threshold).astype(int)

    # The official scorer requires one file per patient.
    for pid, pdf in df.groupby("patient_id"):
        # Ensure sequential timeline
        pdf = pdf.sort_values("ICULOS")

        # Write true label file
        label_path = os.path.join(label_dir, f"{pid}.psv")
        pdf[["label"]].rename(columns={"label": "SepsisLabel"}).to_csv(
            label_path, sep="|", index=False
        )

        # Write prediction file
        pred_path = os.path.join(pred_dir, f"{pid}.psv")
        pdf[["pred_prob", "pred_label"]].rename(
            columns={"pred_prob": "PredictedProbability", "pred_label": "PredictedLabel"}
        ).to_csv(pred_path, sep="|", index=False)

    print(f"Wrote {df.patient_id.nunique()} label files to {label_dir}")
    print(f"Wrote {df.patient_id.nunique()} prediction files to {pred_dir}")
    
    # Print instructions for the user (though our project skips the external repo script now)
    print("\nNext step (run in your terminal):")
    print(f"  python evaluation-2019/evaluate_sepsis_score.py {label_dir} {pred_dir} utility_results.psv")

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")
    fast_mod = import_module("02b_features_fast")
    elig_mod = import_module("08_eligibility_tagging")
    xgb_mod = import_module("04_xgboost_main")

    # Full data pipeline execution
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
                                  
    # Train main model
    model, val_probs = xgb_mod.train_xgb(train_df, val_df, feature_cols)
    
    # Predict on unseen test set
    test_probs = model.predict_proba(test_df[feature_cols])[:, 1]

    # THRESHOLD = 0.80, chosen via threshold sweep in 06_threshold_leadtime.py.
    # It balances early warning rewards against false alarm penalties perfectly.
    write_label_and_prediction_files(
        test_df_with_meta=test_df[["patient_id", "ICULOS", "label"]],
        pred_probs=test_probs,
        threshold=0.80,
        label_dir="utility_eval/labels",
        pred_dir="utility_eval/predictions",
    )