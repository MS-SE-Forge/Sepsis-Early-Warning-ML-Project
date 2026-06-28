# Early Sepsis Warning System — Execution Guide

***

## Prerequisites

- Python **3.9 or higher**
- Jupyter Notebook or JupyterLab
- ~8 GB RAM recommended (dataset is ~1.55 million rows)
- ~5–10 GB free disk space (dataset + artifacts)
- Internet access for first-time package installation (Cell 2.1 auto-installs all dependencies)

***

## Project Structure

```
project-root/
│
├── main.ipynb                  ← Main notebook (the only file you run)
│
├── input/training_setA/              ← Hospital A patient files (.psv)  ← YOU MUST PROVIDE
│   ├── p000001.psv
│   ├── p000002.psv
│   └── ...
│
├── input/training_setB/              ← Hospital B patient files (.psv)  ← YOU MUST PROVIDE
│   ├── p000001.psv
│   ├── p000002.psv
│   └── ...
│
└── artifacts/                  ← Auto-created on first run (do NOT create manually)
    ├── featurecols.json
    ├── valpredictions.csv
    ├── testpredictions.csv
    ├── lr_model.joblib
    ├── xgb_model.joblib
    ├── lgb_model.joblib
    ├── ensemble_weights.joblib
    ├── label_distribution.png
    ├── feature_importance.png
    ├── prcurve_comparison.png
    └── ...
```

***

## Step 1 — Download the Dataset

1. Go to: **https://physionet.org/content/challenge-2019/**
2. Create a free PhysioNet account if you do not have one
3. Download **training_setA.zip** and **training_setB.zip**
4. Unzip both archives into the **same folder as `main.ipynb`**
5. Confirm the folder names are exactly:
   - `training_setA/` — contains ~20,336 `.psv` files
   - `training_setB/` — contains ~20,000 `.psv` files

> **Alternative folder locations supported automatically:**
> The notebook also checks `input/training_setA`, `input/training_setB`, `../training_setA`, and `../training_setB`.
> If none are found, the notebook prints a clear error with instructions.

***

## Step 2 — Install Dependencies

The notebook **auto-installs all dependencies** in **Cell 2.1**. No manual `pip install` is needed.

If you prefer to install manually beforehand:

```bash
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn joblib optuna
```

| Package | Version tested | Purpose |
|---|---|---|
| `xgboost` | ≥ 1.7 | XGBoost classifier |
| `lightgbm` | ≥ 3.3 | LightGBM classifier |
| `scikit-learn` | ≥ 1.2 | Logistic Regression, metrics, splits |
| `pandas` | ≥ 1.5 | Data loading and manipulation |
| `numpy` | ≥ 1.23 | Numerical arrays |
| `matplotlib` | ≥ 3.6 | All plots and figures |
| `seaborn` | ≥ 0.12 | Heatmaps and distribution plots |
| `joblib` | ≥ 1.2 | Saving and loading models |
| `optuna` | ≥ 3.0 | Hyperparameter tuning (Bayesian) |

***

## Step 3 — Open the Notebook

```bash
jupyter notebook main.ipynb
# or
jupyter lab main.ipynb
```

***

## Step 4 — Run the Notebook

### Full first-time run (training included)

Run **all cells from top to bottom** using:
- Menu: **Kernel → Restart & Run All**
- Or press `Shift + Enter` through each cell sequentially

### ⚠️ Important: Training time for Section 5

| Cell | Model | Estimated Time |
|---|---|---|
| Cell 5.2 | Logistic Regression (ElasticNet sweep) | ~15 minutes |
| Cell 5.3 | XGBoost (Optuna 50 trials) | ~25 minutes |
| Cell 5.4 | LightGBM (Optuna 50 trials) | ~15 minutes |
| **Total Section 5** | All models | **~55 minutes** |

> All trained models are saved to `artifacts/`. After the first run, you can **reload artifacts instantly** without retraining — all sections after Section 5 load from disk.

### Subsequent runs (no retraining needed)

If `artifacts/` already exists with trained models:
- Skip Section 5 training cells (5.2, 5.3, 5.4)
- Run from **Section 6 onward** — all results load from saved files in seconds

***

## Step 5 — Notebook Sections at a Glance

| Section | Content | Must Re-run? |
|---|---|---|
| **Section 1** | Problem statement and background | No |
| **Section 2** | Dataset loading, statistics, label distribution plots | Yes (reads raw data) |
| **Section 3** | Manual data inspection — sample rows, missingness | Yes |
| **Section 4** | Full preprocessing pipeline — forward-fill, windowed features, eligibility tagging, train/val/test split | Yes (builds feature matrix) |
| **Section 5** | Model training — qSOFA rule, Logistic Regression, XGBoost, LightGBM, Ensemble | **First time only** (~55 min); saves to artifacts/ |
| **Section 6** | Evaluation metrics on validation and test sets | Yes (loads from artifacts) |
| **Section 7** | Results — tables, PR curves, AUROC curves, feature importance | Yes (loads from artifacts) |
| **Section 8** | Error analysis — 4 case studies (success, miss, false alarm, structural limit) | Yes (loads from artifacts) |
| **Section 9** | Conclusion and limitations | No |

***

## Common Issues and Fixes

| Issue | Cause | Fix |
|---|---|---|
| `Dataset directories NOT found` | `.psv` folders are in wrong location or named differently | Place `training_setA/` and `training_setB/` next to `main.ipynb`; check exact folder names (case-sensitive on Linux/Mac) |
| `ModuleNotFoundError` | Package not installed | Run Cell 2.1 first, or run `pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn joblib optuna` |
| `MemoryError` during feature building | Not enough RAM | Close other applications; feature building uses ~6–8 GB peak RAM |
| Optuna tuning takes too long | 50 trials × 2 models | Reduce `n_trials=50` to `n_trials=20` in Cells 5.3 and 5.4 — models still converge, results may differ slightly |
| `FileNotFoundError` for artifacts | Section 5 was skipped | Run Section 5 training cells at least once to generate `artifacts/` |
| Plots not showing inline | Jupyter display setting | Add `%matplotlib inline` to the first code cell and restart the kernel |

***

## Artifacts Reference

All outputs are saved automatically to `artifacts/`. Do not modify these files manually.

| File | Description |
|---|---|
| `featurecols.json` | List of 81 feature column names used for training |
| `lr_model.joblib` | Trained Logistic Regression pipeline (imputer + scaler + model) |
| `xgb_model.joblib` | Trained XGBoost classifier |
| `lgb_model.joblib` | Trained LightGBM classifier |
| `ensemble_weights.joblib` | Best soft-vote weights `{xgb_w, lgb_w, lr_w}` |
| `valpredictions.csv` | Validation set: patient-hour rows with all model probabilities |
| `testpredictions.csv` | Test set: patient-hour rows with all model probabilities |
| `label_distribution.png` | Patient-level and row-level class imbalance bar chart |
| `feature_importance.png` | XGBoost top-15 feature importance (gain) bar chart |
| `prcurve_comparison.png` | Precision-Recall curves: LR vs XGBoost vs LightGBM vs Ensemble |

***

## Reproducibility

- All random seeds are fixed to **`42`** throughout the notebook
- `GroupShuffleSplit` with `random_state=42` ensures the same train/val/test patients every run
- Optuna uses `TPESampler(seed=42)` for deterministic hyperparameter search
- `np.random.seed(42)` is set at import time

Running the notebook twice on the same machine with the same dataset will produce **identical results**.

***

## Hardware Notes

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| CPU cores | 2 | 8+ (XGBoost and LightGBM use `n_jobs=-1`) |
| Disk space | 5 GB | 10 GB |
| GPU | Not required | Not used |

***

## Dataset Citation

> Reyna M, Josef C, Seyyed M, et al. *Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine, 2020.
> URL: https://physionet.org/content/challenge-2019/