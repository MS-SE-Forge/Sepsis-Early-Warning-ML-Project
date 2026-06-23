"""
Section: Selected Models - Main Model (XGBoost)

This script implements the core predictive model for the Early Sepsis Warning system.

Why XGBoost fits this problem specifically:
1. Handles missing values natively via learned default split directions. We still 
   feed it missingness indicators because the *fact* a value is missing is predictive.
2. Tree-based models cope seamlessly with the vastly different scales of vitals 
   (e.g., HR = 120) vs labs (e.g., pH = 7.3) without needing StandardScaler.
3. `scale_pos_weight` mathematically forces the trees to care about the extreme 
   class imbalance without forcing us to aggressively downsample the dataset.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def train_xgb(train_df, val_df, feature_cols, random_state=42):
    """
    Trains an XGBoost gradient boosted tree model.
    
    Args:
        train_df (pd.DataFrame): The training dataset.
        val_df (pd.DataFrame): The validation dataset for early stopping.
        feature_cols (list): The list of feature columns to pass to the model.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (model, val_probs)
               model (xgb.XGBClassifier): The trained XGBoost model.
               val_probs (np.ndarray): The predicted probabilities on the validation set.
    """
    x_train, y_train = train_df[feature_cols], train_df["label"]
    x_val, y_val = val_df[feature_cols], val_df["label"]

    # Calculate scale_pos_weight = (number of negatives) / (number of positives).
    # This is the standard XGBoost recipe for imbalanced binary classification.
    # It acts as a multiplier for the gradient of the positive class.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Initialize the model with hyperparameters designed to prevent overfitting
    model = xgb.XGBClassifier(
        n_estimators=500,               # Maximum number of trees
        max_depth=5,                    # Keep trees shallow to prevent overfitting on noise
        learning_rate=0.05,             # Slow learning rate for smoother convergence
        subsample=0.8,                  # Use 80% of rows per tree (Stochastic Gradient Boosting)
        colsample_bytree=0.8,           # Use 80% of features per tree (prevents dominating features)
        scale_pos_weight=scale_pos_weight, # Address extreme class imbalance
        eval_metric="aucpr",            # Crucial: Optimize toward Precision-Recall, NOT accuracy/AUC
        early_stopping_rounds=30,       # Stop adding trees if validation AUPRC hasn't improved in 30 rounds
        random_state=random_state,
        n_jobs=-1,                      # Use all CPU cores
    )
    
    # Train the model, monitoring performance on the validation set
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False # Keep terminal output clean during automated runs
    )

    # Extract the probability that the patient will develop sepsis
    val_probs = model.predict_proba(x_val)[:, 1]
    
    return model, val_probs

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("eda")
    feat_mod = import_module("features_split")

    # Load and prep data
    df = eda.load_all_patients(eda.DATA_DIRS)
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = feat_mod.build_windowed_features(df)
    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    # Filter out metadata columns so XGBoost only sees math features
    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source", "label")]

    # ALWAYS sanity check your validation split class counts. If validation has 0 positives,
    # AUPRC optimization will crash.
    n_pos_val = val_df["label"].sum()
    print(f"Validation set: {len(val_df)} rows, {n_pos_val} positive "
          f"({n_pos_val/len(val_df)*100:.2f}%) -- ALWAYS check this raw count, "
          f"not just the percentage, before trusting AUPRC/early-stopping signals.\n")

    # Train and evaluate
    model, val_probs = train_xgb(train_df, val_df, feature_cols)
    print("=== XGBoost main model (val set) ===")
    print(f"AUROC: {roc_auc_score(val_df['label'], val_probs):.3f}")
    print(f"AUPRC: {average_precision_score(val_df['label'], val_probs):.3f}")
    
    # Show exactly where early stopping kicked in (the optimal number of trees)
    print(f"Best iteration (early stopping): {model.best_iteration}")

    # Extract feature importances to see what clinical variables XGBoost relies on most
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))
