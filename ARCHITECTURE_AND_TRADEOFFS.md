# Comprehensive Project Documentation & Architecture Guide

Welcome to the Early Sepsis Warning project. This document outlines the architectural decisions and clinical trade-offs made during the development of this codebase.

---

## 1. System Architecture & Codebase Flow

The project is structured as a sequential pipeline, explicitly numbered from `01` to `12`. This modular approach ensures that each script has a single responsibility.

1. **Ingestion & EDA (`eda.py`, `onset_window_diagnostic.py`)**: 
   - Loads the raw `.psv` files.
   - Calculates baseline statistics (missingness, label balance).
2. **Feature Engineering & Leakage Prevention (`features_split.py`, `features_fast.py`, `eligibility_tagging.py`)**:
   - Imputes missing data using clinical logic.
   - Extracts 6-hour rolling window features (means, slopes).
   - Segregates patients by eligibility (can they even be caught early?).
   - Performs a strict patient-level Train/Val/Test split.
3. **Modeling (`baselines.py`, `xgboost_main.py`)**:
   - Trains the primary XGBoost classifier, optimized for `aucpr` to handle extreme class imbalance.
4. **Evaluation & Clinical Utility (`05`, `06`, `09`, `10`, `11`)**:
   - Sweeps for the optimal probability threshold based on an *approximate utility score* (rewarding lead time, penalizing false alarms).
   - Stratifies results by hospital and patient subgroup.
   - Extracts specific error analysis examples (Success, Miss, False Alarm).
5. **Orchestration & Reporting (`build_artifacts.py`, `build_notebook.py`)**:
   - The master script (`12`) runs everything end-to-end and saves the model and metric tables to an `artifacts/` directory.
   - The report builder takes those static artifacts and builds `sepsis_early_warning.ipynb` for instant analysis without slow retraining.

---

## 2. Key Trade-off Decisions and Why

### A. XGBoost vs. Deep Learning (e.g., LSTM / RNN)
*   **The Decision**: We chose XGBoost combined with manual feature engineering over a Deep Learning sequence model.
*   **The Why**: XGBoost trains in minutes, is highly interpretable (crucial for clinical interpretability), handles `NaN` values natively, and requires far less tuning. 
*   **The Trade-off**: We had to manually construct sequence features (6-hour rolling means and slopes). An LSTM would learn these temporal patterns automatically, but would be a black box, harder to train, and harder to justify clinically.

### B. Forward Filling vs. Mean Imputation
*   **The Decision**: We use `ffill()` (forward fill) within a patient's timeline for missing data, rather than filling all missing gaps with the population average.
*   **The Why**: This reflects clinical reality. If a doctor draws a Lactate lab and it reads 2.5, they treat the patient as having a Lactate of 2.5 for the next several hours until a new lab is drawn. 
*   **The Trade-off**: At the very beginning of an ICU stay, there are no "past" values to forward-fill. We are forced to use the population median for the first few hours until the first labs are drawn.

### C. Patient-Level Split vs. Random Row Split
*   **The Decision**: We enforce a strict `GroupShuffleSplit` grouping by `patient_id`.
*   **The Why**: **Leakage Prevention.** If we randomly shuffled all 1.5 million hours, Patient A's hour 12 might end up in the training set, while Patient A's hour 8 ends up in the test set. The model would effectively look into the future, completely invalidating the test metrics.
*   **The Trade-off**: The distributions of the Train and Test sets have slightly higher variance because patients have vastly different lengths of stay (one patient might provide 20 hours, another might provide 200). 

### D. Utility Thresholding vs. Standard Metrics (F1 / ROC-AUC)
*   **The Decision**: Instead of using a standard `0.5` probability threshold to declare a patient "septic", we sweep thresholds and select `0.80` based on a custom utility function.
*   **The Why**: Standard metrics don't care about "Lead Time". Our utility function actively rewards the model for flagging sepsis *hours before it happens*, while heavily penalizing continuous false alarms (alarm fatigue). 
*   **The Trade-off**: By optimizing for utility, we sacrifice raw Recall (Sensitivity). We miss some patients entirely, but the ones we do catch are caught early, and we don't bombard nurses with false alarms.

### E. Explicit Missingness Indicators
*   **The Decision**: We created binary columns like `Lactate_missing`.
*   **The Why**: In a hospital, data is not "Missing at Random". The very act of a doctor *ordering* a specific lab means they suspect something is wrong. The absence or presence of a measurement is a predictive feature in itself.

### F. Robustness & Distribution Shift (Hospital A vs. Hospital B)
*   **The Decision**: We explicitly evaluate the model's performance stratified by hospital source (`hospital_subgroup.py`).
*   **The Why**: Evaluating robustness against **distribution shift** is critical. A model that memorizes the clinical protocols of Hospital A might fail catastrophically at Hospital B. By evaluating subgroups separately, we ensure that our model generalizes across different healthcare systems, not just a single dataset.

### G. Artifact Pre-computation Strategy
*   **The Decision**: We decoupled the heavy training pipeline (`build_artifacts.py`) from the reporting layer (`build_notebook.py`).
*   **The Why**: Training an XGBoost model on 1.5 million rows takes several minutes. By saving the model, predictions, and a small subset of "demo patients" to the `artifacts/` folder, the Jupyter notebook analysis loads and predicts instantly.

---

## 3. Known Limitations

1. **The "Immediate Only" Subgroup**: About 25% of the septic patients in the dataset arrive at the ICU and are labeled septic within the first 3 hours. It is **mathematically impossible** for an algorithm to provide a 6-hour early warning for these patients. Our pipeline explicitly tags these patients so we don't unfairly penalize the model for failing an impossible task.
2. **Treatment Bias**: We do not have data on the interventions doctors performed (e.g., when antibiotics were administered). A model might flag a patient, a doctor gives antibiotics, and the patient never gets sepsis. In our dataset, the model would be penalized for a "False Alarm", even though it actually saved the patient's life.
3. **Data Sparsity**: Vitals are taken hourly, but labs are taken rarely. The 6-hour window often relies heavily on stale lab data.

---

## 4. Possible Future Issues & Enhancements

*   **Model Drift & Calibration**: If hospital protocols change (e.g., a new policy requires Lactate tests every 2 hours instead of 6), the model's reliance on missingness indicators will break. The model must be periodically recalibrated.
*   **Deployment Complexity**: Running this in a real hospital requires an active, streaming data pipeline that aggregates EHR (Electronic Health Record) data every hour. Calculating rolling OLS slopes in a real-time Kafka stream is significantly harder than doing it in a static pandas DataFrame.
*   **Generalizability (External Validation)**: The model is trained on PhysioNet 2019 data (two hospitals). If deployed to a rural clinic with completely different baseline patient demographics and different lab equipment calibration, the model's performance could drop drastically. Future work must validate this model on an external, out-of-distribution dataset (like MIMIC-IV).
