# Early Sepsis Warning with Utility-Based Clinical Thresholds

## Project Overview

This repository contains the codebase for our Machine Learning Project: an Early Sepsis Warning System. The core objective is to predict sepsis onset in ICU patients *before* it becomes clinically obvious, providing actionable lead time for medical intervention without overwhelming staff with false alarms.

Instead of relying on standard binary classification metrics like Accuracy or AUROC (which are misleading under the severe class imbalances of clinical data), this project optimizes for **clinical utility**, penalizing false alarms and missed cases while explicitly rewarding early true positives.

## Dataset

We use the **PhysioNet / Computing in Cardiology Challenge 2019** dataset. 
*(Note: To comply with data sharing rules and repository size limits, the raw dataset is not included in this repository. You must download it separately).*

1. Download `training_setA` and `training_setB` from [PhysioNet 2019](https://physionet.org/content/challenge-2019/1.0.0/).
2. Place the downloaded `.psv` files into a `physionet.org/` directory in the root of this project.

## Installation & Setup

1. Clone this repository.
2. Ensure you have Python 3.9+ installed.
3. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Execution

The project is structured as a sequential pipeline. While you can run individual scripts (`01_eda.py`, `02_features_split.py`, etc.) for debugging, the easiest way to reproduce the entire project end-to-end is via the master build script:

```bash
python 12_build_artifacts.py
```

This will:
1. Load and clean the dataset.
2. Forward-fill vitals/labs and extract 6-hour sliding window features.
3. Apply a patient-level `GroupShuffleSplit` to prevent leakage.
4. Train the primary XGBoost model, handling class imbalance natively.
5. Compute validation metrics (including the utility proxy and lead-time analysis).
6. Save the trained model, metrics, and demo datasets into the `artifacts/` folder.

## Final Presentation

Once the artifacts are built, you can generate the final Jupyter Notebook presentation:

```bash
python build_notebook.py
```

This generates `sepsis_early_warning.ipynb`, which contains a complete walkthrough of our methodology, interactive data exploration, error analysis visualizations, and a live demonstration of the model using the pre-computed artifacts.

## Architecture & Design Choices

*   **Windowing**: We use a 6-hour sliding window to capture trends (slopes and moving averages) of vital signs, recognizing that clinical data is a time-series, not a static snapshot.
*   **Leakage Prevention**: We enforce strict split boundaries based on `patient_id`. Random row-level splitting causes severe data leakage in clinical datasets.
*   **Missingness**: Missing lab values carry implicit clinical signals (e.g., a lab wasn't ordered). We use missingness indicator variables and forward-filling rather than mean-imputation.
*   **Early Warning Tagging**: Patients who are septic upon ICU admission structurally cannot be "predicted early". Our pipeline explicitly tags these patients (`immediate_only`) and stratifies evaluation results to accurately reflect true early-warning capabilities.
