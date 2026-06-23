"""
Section: Selected Models - Main Model (XGBoost)

Why XGBoost fits this problem specifically:
- Handles missing values internally via learned default split direction,
  so we don't strictly need our forward-fill for XGBoost's own splits
  (though we keep it + missingness indicators since they still add
  genuine signal: "how long ago was this last measured").
- Tree-based models cope well with the very different scales/distributions
  of vitals vs labs without needing careful feature scaling.
- Fast enough to train on the full dataset and to re-tune iteratively.
- scale_pos_weight directly addresses class imbalance documented in EDA.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

def train_xgb(train_df, val_df, feature_cols, random_state=42):
    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_val, y_val = val_df[feature_cols], val_df["label"]

    # scale_pos_weight = (negatives / positives), the standard XGBoost
    # recipe for imbalanced binary classification
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # optimize toward the imbalance-aware metric
        early_stopping_rounds=30,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    return model, val_probs

if __name__ == "__main__":
    from importlib import import_module
    eda = import_module("01_eda")
    feat_mod = import_module("02_features_split")

    df = eda.load_all_patients(["training_setA"])
    df = feat_mod.add_missingness_indicators(df, feat_mod.LABS)
    df = feat_mod.forward_fill_within_patient(df, feat_mod.VITALS + feat_mod.LABS)
    feat_df = feat_mod.build_windowed_features(df)
    train_df, val_df, test_df = feat_mod.patient_level_split(feat_df)

    feature_cols = [c for c in train_df.columns
                     if c not in ("patient_id", "ICULOS", "hospital_source", "label")]

    n_pos_val = val_df["label"].sum()
    print(f"Validation set: {len(val_df)} rows, {n_pos_val} positive "
          f"({n_pos_val/len(val_df)*100:.2f}%) -- ALWAYS check this raw count, "
          f"not just the percentage, before trusting AUPRC/early-stopping signals.\n")

    model, val_probs = train_xgb(train_df, val_df, feature_cols)
    print("=== XGBoost main model (val set) ===")
    print(f"AUROC: {roc_auc_score(val_df['label'], val_probs):.3f}")
    print(f"AUPRC: {average_precision_score(val_df['label'], val_probs):.3f}")
    print(f"Best iteration (early stopping): {model.best_iteration}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))
