"""
Section: Selected Models - Baselines

This script establishes two critical baselines required to contextualize performance:
1. Clinical-rule reference: A sanity check using a qSOFA-inspired hardcoded rule.
   This proves whether machine learning is actually better than a simple clinical heuristic.
2. Logistic regression: A linear, learned baseline to prove whether the complexity 
   of tree-based models (XGBoost) is justified over a simpler, more interpretable model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def clinical_rule_score(feature_df):
    """
    Applies a simple clinical rule inspired by qSOFA (quick Sequential Organ Failure Assessment).
    
    True qSOFA requires a mentation/GCS check, which this dataset lacks. 
    We adapt it using only available vitals. If a patient meets the criteria, 
    they get a point. Score >= 1 flags the patient for potential sepsis.
    
    This is NOT a trained model. It's a heuristic baseline.
    
    Args:
        feature_df (pd.DataFrame): The feature dataset.
        
    Returns:
        tuple: (score, flagged) 
               score (pd.Series): The raw heuristic score (0 to 2).
               flagged (pd.Series): Binary indicator if score >= 1.
    """
    # Award 1 point if Respiratory Rate is elevated (tachypnea)
    # Award 1 point if Systolic Blood Pressure is low (hypotension)
    score = (feature_df["Resp_last"] >= 22).astype(int) + \
            (feature_df["SBP_last"] <= 100).astype(int)
            
    # Treat a score of 1 or higher as a positive "flag" for baseline comparison
    flagged = (score >= 1).astype(int)
    
    return score, flagged

def train_logistic_baseline(train_df, val_df, feature_cols):
    """
    Trains a Logistic Regression model to serve as the machine-learned baseline.
    
    CRITICAL: Because Logistic Regression is distance-based, features MUST be 
    scaled. Also, because sepsis is extremely imbalanced (~1.8% of hours), 
    we MUST use class_weight='balanced' so the model doesn't just trivially 
    predict '0' every time to achieve 98% accuracy.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
        feature_cols (list): List of column names to use as features.
        
    Returns:
        tuple: (pipe, val_probs)
               pipe (sklearn.Pipeline): The trained scaler + logistic regression model.
               val_probs (np.ndarray): Predicted probability of sepsis for the validation set.
    """
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]

    # Construct a pipeline to ensure scaling happens strictly on training data
    # (preventing data leakage of mean/variance from the validation set).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", # heavily penalize missing the rare class
            max_iter=2000,           # allow enough iterations for convergence
            random_state=42
        ))
    ])
    
    # Train the linear baseline
    pipe.fit(X_train, y_train)

    # Extract probabilities for the positive class (sepsis)
    val_probs = pipe.predict_proba(X_val)[:, 1]
    
    return pipe, val_probs

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")

    # Standard data loading pipeline
    df = eda.load_all_patients(["training_setA"])
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = feat_mod.build_windowed_features(df)
    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    # 1. Evaluate the Clinical Rule Baseline
    score, flagged = clinical_rule_score(val_df)
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("=== Clinical rule reference (val set) ===")
    print(f"Flag rate: {flagged.mean()*100:.2f}%")
    print(f"AUROC (using raw 0/1/2 score as a ranking): "
          f"{roc_auc_score(val_df['label'], score):.3f}")

    # 2. Evaluate the Logistic Regression Baseline
    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source", "label")]
    
    pipe, val_probs = train_logistic_baseline(train_df, val_df, feature_cols)
    print("\n=== Logistic Regression baseline (val set) ===")
    print(f"AUROC: {roc_auc_score(val_df['label'], val_probs):.3f}")
    
    # AUPRC (Area Under Precision-Recall Curve) is far more informative than AUROC 
    # for highly imbalanced datasets.
    print(f"AUPRC: {average_precision_score(val_df['label'], val_probs):.3f}")
