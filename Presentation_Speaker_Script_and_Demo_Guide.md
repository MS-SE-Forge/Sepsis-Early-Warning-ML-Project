# 🎙️ Master Team Presentation Talk & Speaker Coaching Guide (Cell-by-Cell Annotated)

This document is written as a **complete, word-for-word spoken script and cell-by-cell explanation guide** designed to help every team member explain their section with absolute mastery and confidence. 

Instead of memorizing dry bullet points, read through your conversational script and refer to the **Cell-by-Cell Commentary Guide** under your section. It explains **why** we made each decision in plain, natural language so you sound like an experienced Machine Learning Engineer during the talk and Q&A.

---

## ⏱️ Presentation Schedule & Roles

| Team Member | Allocated Time | Core Topics Owned | Exact Notebook Cells to Display & Explain |
|---|---|---|---|
| **Speaker 1** | 0:00 — 1:15 | Introduction, Sepsis Problem & Dataset Ingestion | **Cell 0** (Title) → **Cells 2.1 to 2.4** (Setup & Ingestion) → **Cells 2.5 to 2.7** (EDA Charts) |
| **Speaker 2** | 1:15 — 2:30 | Clinical Preprocessing, Imputation & 6h Windows | **Cells 4.1 to 4.5** (Imputation) → **Cell 4.6** (6h Windows) → **Cells 4.7 to 4.9** (Split & Assert) |
| **Speaker 3** | 2:30 — 3:45 | Model Architecture, Optuna Tuning & Threshold Sweep | **Cells 5.1 to 5.6** (Models & Ensemble) → **Cells 6.1 to 6.6** (Utility Scorer & Cutoff Sweep) |
| **Speaker 4** | 3:45 — 5:00 | Test Results, Error Case Studies & Live Streaming Demo | **Section 7** (Test Results) → **Cells 8.1 to 8.4** (Error Cases) → **Cells 10.1 to 10.5** (Live Demo) |

---

## 📢 Detailed Spoken Script & Cell-by-Cell Explanations

### 👤 Speaker 1: The Medical Problem & Data Landscape (0:00 — 1:15)

#### 🎬 Stage Cue:
Open the notebook at **Cell 0 (Title Slide)**. As you start speaking, smoothly scroll down through **Cells 2.1 to 2.4** showing data loading, and then pause on **Cells 2.5 to 2.7** so the judges see the label distribution and lab missingness charts.

#### 🗣️ Word-for-Word Spoken Script:
> "Good morning respected judges, professors, and guests. We are Team ML_SE_ALL_02, and we are proud to present our applied machine learning project: **Early Sepsis Warning with Utility-Based Clinical Thresholds**.
> 
> *(Point to Cell 0)*
> Starting right here at **Cell 0**, our research focuses on Intensive Care Units where sepsis—a deadly organ collapse caused by infection—is a race against the clock. Every hour antibiotic treatment is delayed increases mortality by 7%. But why not just alarm at the slightest fever? Because of 'Alarm Fatigue.' Up to 90% of hospital bedside alerts are false alarms, causing doctors to subconsciously ignore monitor beepings. We need an AI that detects sepsis **hours before clinical onset**, while keeping false alarms strictly penalized.
> 
> *(Scroll through Cells 2.1, 2.2, 2.2b, and 3.0)*
> In **Cell 2.1 and 2.2**, we install and import our specialized stack: `xgboost`, `lightgbm`, `optuna`, and `joblib`. Notice **Cell 2.2b**: we built an intelligent master gate called `FORCE_RETRAIN`. Set to `False`, it triggers **Cell 3.0**, our Centralized Artifact Loader, which pre-loads pre-trained models into memory in 60 seconds for live demos. If set to `True`, it runs our 7-minute training pipeline.
> 
> *(Scroll to Cell 2.3 and 2.4)*
> In **Cell 2.3 and 2.4**, our `load_all_patients` function ingests the official PhysioNet 2019 dataset across two hospital systems. As you can see printed from **Cell 2.4**, we successfully ingested **40,336 patient stays**, representing over **1.55 million hourly vital sign records**.
> 
> *(Pause on Cells 2.5, 2.6, and 2.7)*
> Our EDA plots in **Cells 2.5 through 2.7** reveal our two major modeling hurdles: First, extreme class imbalance—only **1.8%** of rows are septic. Second, lab tests like Lactate are missing over 90% of the time. I will now hand over to Member 2 to explain how our preprocessing transforms these hurdles into predictive power."

#### 🔬 Speaker 1 Cell-by-Cell Commentary Guide:
* **Cell 0 (Title Block)**: Displays project metadata, team members, matriculation numbers, and deadline. Establish professional context here.
* **Cell 2.1 (Dependencies Installation)**: Executes a quiet `pip install` for tree boosting (`xgboost`, `lightgbm`), hyperparameter tuning (`optuna`), and serialization (`joblib`).
* **Cell 2.2 (Core Imports)**: Imports data manipulation (`pandas`, `numpy`), partitioning (`scikit-learn`), and plotting (`matplotlib`).
* **Cell 2.2b (Master Run Gate)**: Sets `FORCE_RETRAIN = False`. Checks disk for all 8 required `.joblib` model/prediction files.
* **Cell 3.0 (Centralized Artifact Loader)**: Executes when `should_train = False`. Loads feature definitions, trained models, validation/test predictions, and the optimal threshold into memory instantly.
* **Cell 2.3 (`load_all_patients` function)**: Scans directories for `.psv` files, extracts patient IDs from file names, attaches hospital origin tags (`A` vs `B`), and concatenates them.
* **Cell 2.4 (Execution & Summary Stats)**: Calls `load_all_patients()`. Prints summary stats: **40,336 patients**, **1,552,210 rows**, overall sepsis rate **1.8%**.
* **Cells 2.5 to 2.7 (EDA Charts)**: Plots the label distribution pie chart (`label_distribution.png`), lab missingness bar chart (`missingness.png`), and sample vital sign trajectories.

---

### 👤 Speaker 2: Preprocessing, 6h Sliding Windows & Leakage Prevention (1:15 — 2:30)

#### 🎬 Stage Cue:
Scroll smoothly through **Cells 4.1 to 4.5**. Spend significant time explaining **Cell 4.6 (`build_windowed_features`)**. Finally, point directly to **Cell 4.8 and Cell 4.9**, highlighting the green assertion printout confirming `0 overlapping patients`.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 1. To process 1.5 million raw hospital records without corrupting our models, we designed a robust clinical preprocessing pipeline in **Section 4**.
> 
> *(Scroll through Cells 4.1 to 4.5)*
> In **Cell 4.1 and 4.2**, we group variables and tag patient eligibility. How do we handle missing vital signs? Standard textbooks suggest mean imputation—replacing missing values with the hospital average. In an ICU, that is fatal! If a patient's heart rate drops to 40, replacing a missing reading an hour later with the hospital average of 80 masks their collapse! In **Cell 4.4**, we implement **Clinical Forward-Filling**: carrying the last measured reading forward, reflecting clinical reality until re-tested. Remaining start-of-stay gaps are median-filled in **Cell 4.5**. In **Cell 4.3**, we create binary missingness indicators to track test-ordering frequency.
> 
> *(Point to Cell 4.6)*
> Doctors don't diagnose patients by looking at an isolated single hour; they look at momentum. In **Cell 4.6**, our `build_windowed_features` engine constructs **6-hour sliding windows**. For every vital sign at hour $T \ge 6$, we slice the past 6 hours, extract the current value (`_last`), rolling average (`_mean`), and fit linear regression to compute the **slope (`_slope`)**. A rising heart rate slope captures acute deterioration long before static thresholds break! We also calculate 6-hour lab missingness rates (`_miss_rate`), letting the AI learn when doctors cluster blood tests.
> 
> *(Point to Cell 4.8 and 4.9)*
> Finally, we prevented the deadliest trap in healthcare AI: **Time-Series Leakage**. Standard random splits scatter hours from the same patient across train and test sets, causing models to memorize patient identity. In **Cell 4.8**, we execute a strict **Patient-Level Split** using `GroupShuffleSplit` on unique `patient_id`s (70% Train, 10% Val, 20% Test). Right here in **Cell 4.9**, our automated assertion confirms **0 overlapping patients** across splits. I now turn it over to Member 3 for modeling."

#### 🔬 Speaker 2 Cell-by-Cell Commentary Guide:
* **Cell 4.1 (Feature Categorization)**: Groups arrays into `VITALS` (8 columns), `LABS` (26 columns), and `DEMOGRAPHICS` (6 columns).
* **Cell 4.2 (Eligibility Tagging)**: Classifies patients into `never_septic`, `early_warning_eligible` (sepsis onset ≥ hour 7), and `immediate_only` (sepsis onset within first 6 hours).
* **Cell 4.3 (Missingness Indicators)**: Creates binary flags (`1` if measured, `0` if NaN) to capture doctor test-ordering frequency.
* **Cell 4.4 (Forward-Fill Imputation)**: Carries each patient's last measured vital sign forward hour-by-hour using `.groupby('patient_id').ffill()`.
* **Cell 4.5 (Median Imputation)**: Imputes remaining start-of-stay NaNs using population median.
* **Cell 4.6 (`build_windowed_features`)**: For every hour $T \ge 6$, slices the past 6 hours. Computes `_last`, rolling 6h `_mean`, linear regression `_slope` via `np.polyfit`, and lab `_miss_rate`. Drops hours 1–5 per patient.
* **Cell 4.7**: Attaches eligibility labels to the engineered feature matrix.
* **Cell 4.8 (`patient_level_split`)**: Uses `GroupShuffleSplit` on unique `patient_id`s to divide data into **Train (70%)**, **Validation (10%)**, and **Test (20%)**.
* **Cell 4.9 (Leakage Assertion)**: Runs automated Python `assert` verifying set intersection between train/val/test patient IDs equals `0`.

---

### 👤 Speaker 3: Model Architecture & Threshold Calibration (2:30 — 3:45)

#### 🎬 Stage Cue:
Scroll through **Cells 5.1 to 5.5b**, explaining the model lineup. Then display **Cell 6.1 (PhysioNet Scorer)**. Finally, point directly to **Cell 6.4 and Cell 6.6**, showing the **Threshold Sweep Graph (`threshold_sweep.png`)** peaking at 0.34.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 2. In **Section 5**, we evaluate our machine learning models against clinical reality.
> 
> *(Scroll through Cells 5.1 to 5.5b)*
> We first established a baseline in **Cell 5.1** using the hospital **qSOFA rule**—the manual respiration and blood pressure checklist. It achieved a tiny validation utility of `0.005`. Next, we trained three diverse algorithms: regularized ElasticNet Logistic Regression in **Cell 5.2**, gradient boosted XGBoost tuned across 50 Optuna trials in **Cell 5.3**, and LightGBM tuned across 50 trials in **Cell 5.4**. In **Cell 5.3**, we explicitly set `scale_pos_weight` to force trees to heavily penalize missed sepsis alarms. In **Cell 5.5b**, grid search discovered our optimal **Soft-Vote Ensemble blend**: 45% XGBoost, 45% LightGBM, and 10% Logistic Regression.
> 
> *(Point to Cell 6.1)*
> How do we grade these models? Standard accuracy is misleading; predicting 'No Sepsis' for everyone yields 98.2% accuracy! In **Cell 6.1**, we embedded the official verbatim **PhysioNet Utility Scorer**. It awards maximum reward (**+1.0 point**) for alarms sounding 6 to 12 hours before clinical onset, drops toward zero for late alerts, and actively penalizes false alarms (**-0.05 points**).
> 
> *(Point to Cell 6.4 and 6.6)*
> This leads to our primary technical breakthrough in **Cell 6.4: Threshold Calibration**. Standard classifiers trigger at probability 0.50. Under 1.8% class imbalance, models rarely reach 0.50 confidence, causing massive miss rates. We ran a systematic threshold sweep from 0.05 to 0.95 on validation data. As shown in **Cell 6.6**, the default 0.50 cutoff yields a poor utility of `0.08`. But lowering our decision cutoff to **0.34** captures deteriorating patients early while controlling false alarms, maximizing validation utility at **0.341**—over 60 times higher than qSOFA! Member 4 will now present our final test demonstration."

#### 🔬 Speaker 3 Cell-by-Cell Commentary Guide:
* **Cell 5.1 (qSOFA Clinical Baseline)**: Computes manual clinical baseline ($+1$ if Resp $> 22$, $+1$ if SBP $< 100$). Validation Utility: `0.005`.
* **Cell 5.2 (Logistic Regression)**: Trains ElasticNet regularized linear model. Saves `lr_model.joblib`.
* **Cell 5.3 (XGBoost + Optuna)**: Runs 50 Bayesian tuning trials optimizing max depth, learning rate, and child weight. Sets `scale_pos_weight` to combat imbalance. Saves `xgb_model.joblib`.
* **Cell 5.4 (LightGBM + Optuna)**: Runs 50 tuning trials for leaf-wise gradient boosting. Saves `lgb_model.joblib`.
* **Cell 5.5b (Soft-Vote Ensemble)**: Grid searches probability weights. Discovers optimal mix: **0.45 XGB + 0.45 LGB + 0.10 LR**. Saves `ensemble_weights.joblib`.
* **Cell 5.6 (Feature Importances)**: Plots Top 15 XGBoost feature importances (highlighting rolling slopes and lab missingness).
* **Cell 6.1 (PhysioNet Scorer)**: Verbatim official reward function (+1.0 early warning 12h-6h prior, -0.05 false alarm penalty).
* **Cell 6.2**: Fast-loads validation/test prediction probabilities.
* **Cell 6.3**: Plots the official PhysioNet reward timeline curve.
* **Cell 6.4 (Threshold Calibration Sweep)**: Sweeps decision cutoffs from `0.05` to `0.95` on Validation set. Proves 0.50 fails (`0.08`) and discovers optimal clinical cutoff at **P ≥ 0.34** (`0.341`).
* **Cells 6.5 & 6.6**: Displays model comparison summary table and overlays validation sweep graphs.

---

### 👤 Speaker 4: Results, Error Analysis & Live Demo (3:45 — 5:00)

#### 🎬 Stage Cue:
Scroll to **Section 7 (Test Evaluation)** and highlight the 0.327 Test Score. Walk through **Cells 8.1 to 8.4 (Error Case Studies)**. Finally, execute or display **Cells 10.1 to 10.5**, pointing to the live alarm firing at Hour 18.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 3. In **Section 7**, we locked our calibrated 0.34 threshold and evaluated the Ensemble on the completely unseen 20% Test set. 
> 
> *(Point to Section 7 Results Table)*
> Our model achieved an official Test Clinical Utility Score of **0.327** and an AUPRC of **0.098**—outperforming the hospital qSOFA baseline by **over 60-fold**! Most importantly, our system gave doctors a median advance warning lead time of **6.0 full hours** before clinical onset. Subgroup breakdown confirmed identical performance across Hospital A and Hospital B.
> 
> *(Scroll through Cells 8.1 to 8.4)*
> In **Cells 8.1 through 8.4**, we performed clinical error analysis across 4 patient case studies: Success, Miss, False Alarm, and Structural Limitation. We uncovered a critical clinical insight: roughly **25% of septic patients arrive at the ICU already septic on Hour 1**. It is a structural impossibility for any monitor to give a 6-hour advance warning for a patient arriving in septic shock! Being honest about this structural limitation explains why the maximum theoretical utility ceiling in ICUs is around 0.65, not 1.0. **Cell 9.1** summarizes these conclusions neatly.
> 
> *(Walk through Cells 10.1 to 10.5)*
> Finally, we bring our system to life in **Section 10**. Watch as our inference engine streams an unseen test patient's vitals hour-by-hour. Notice **Cells 10.3 and 10.4**: the patient appears stable, but rising heart rate slopes and falling blood pressure push our ensemble probability across the 0.34 threshold at ICU Hour 18. The system sounds a high-priority early warning alarm **exactly 6 hours before** clinical sepsis onset occurs at Hour 24! 
> 
> This concludes our presentation. Our pipeline is verified, interpretable, and ready for deployment testing. Thank you, and we welcome your questions."

#### 🔬 Speaker 4 Cell-by-Cell Commentary Guide:
* **Section 7 (Test Evaluation)**: Evaluates ensemble on unseen Test set at `0.34` cutoff. Official Test Utility: **`0.327`** (60x higher than qSOFA). Outputs hospital subgroup table.
* **Cells 8.1 to 8.4 (Error Case Studies)**: Plots 4 distinct patient trajectories: Success (early alert), Miss (blunted fever), False Alarm (hypertension), and Structural Limit (patient arrives septic on Hour 1).
* **Cell 9.1 (Conclusion Table)**: Summarizes primary research findings answering core questions.
* **Cells 10.1 to 10.5 (Live Demo Walkthrough)**: Loads saved models fresh. Streams an unseen test patient's hourly vital signs. Shows ensemble probability climbing across `0.34` threshold at Hour 18, triggering a live alert **6 hours early**.

---

## 🛡️ Master Team Q&A Coaching Guide

When answering questions, **do not give one-word answers**. Use the proven explanations below:

| Professor's Question | Winning Team Explanation |
|---|---|
| **"Why is your AUPRC score around 0.10? Isn't 0.10 a terrible score?"** | "In standard balanced datasets, yes, but not under extreme 1.8% class imbalance! In AUPRC, a random guess scores equal to positive class prevalence—which is 0.018. Our score of 0.098 represents more than a **5-fold improvement over random baseline**, and is 5 times higher than the hospital qSOFA score (0.019)." |
| **"How do we know your Optuna tuning didn't just overfit to validation data?"** | "We prevented overfitting through three layers: First, 5-fold cross-validation inside Optuna with early stopping. Second, strict separation of patients using GroupShuffleSplit. And third, empirical proof: our Utility score held completely stable on the unseen 20% Test set (**0.327 Test vs 0.341 Validation**)." |
| **"What happens if I change `FORCE_RETRAIN = True` and execute right now?"** | "The centralized loader will bypass disk loading and trigger the full pipeline: it will load raw `.psv` files, compute forward fills and 6h rolling slopes, run 50 Optuna optimization trials per boosted tree model, search grid weights for the ensemble, re-calibrate the 0.34 threshold, and overwrite the saved artifacts. It completes smoothly in about 7 minutes." |
| **"Why did you include `immediate_only` patients if you know you can't predict them early?"** | "To preserve strict data integrity and clinical transparency. Filtering out patients who arrive already septic would artificially inflate our score to look better than reality. Instead, we report overall results honestly and explain the ~25% structural admission limitation in our error analysis section." |
