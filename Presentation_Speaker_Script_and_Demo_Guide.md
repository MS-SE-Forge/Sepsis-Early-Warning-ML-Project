# 🎙️ Master 6–7 Minute Presentation Guide & Speaker Script (Cell-by-Cell Annotated)

This document is tailored strictly to your class presentation requirements: **6–7 minutes total runtime** (~1.5 minutes per speaker). Introductory background has been streamlined so you immediately present the core pillars: **Problem Solved**, **Technical Requirements**, **Methodology Steps Followed**, and **Results & Findings**. 

Every single visual figure, output statistic, and code block from **Cell 0 to Cell 10.5** is explicitly accounted for in both the spoken scripts and the underlying commentary tables.

---

## ⏱️ Presentation Schedule & Core Pillars (6–7 Minutes Total)

| Team Member | Allocated Time | Core Pillar Focus | Exact Notebook Cells & Figures Displayed |
|---|---|---|---|
| **Speaker 1** | 0:00 — 1:30 | **The Problem & Requirements**<br/>Defines clinical utility goals, dataset ingestion, and bottlenecks. | **Cell 0** (Title) → **Cells 2.1–2.2b** (Gate) → **Cell 3.0** (Loader) → **Cells 2.3–2.4** (Stats) → **Cells 2.5–2.7** (EDA Figures) |
| **Speaker 2** | 1:30 — 3:00 | **Steps Followed: Preprocessing**<br/>Explains clinical imputation, 6h sliding windows, and leakage prevention. | **Section 3** (Roadmap) → **Cells 4.1–4.5** (Imputation) → **Cell 4.6** (6h Windows) → **Cells 4.7–4.9** (Split & Assert Output) |
| **Speaker 3** | 3:00 — 4:30 | **Steps Followed: Modeling & Calibration**<br/>Covers Optuna tuning, 45/45/10 ensemble blending, and threshold sweep. | **Cells 5.1–5.6** (Models & Importances) → **Cells 6.1–6.3** (Scorer) → **Cells 6.4–6.6** (Threshold Sweep Figure) |
| **Speaker 4** | 4:30 — 6:15 | **Results & Findings**<br/>Presents 60x utility gain, 4 error trajectories, structural limits, and live demo. | **Section 7** (Test Table) → **Cells 8.1–8.4** (Error Trajectories) → **Cell 9.1** (Summary) → **Cells 10.1–10.5** (Live Stream Demo) |

---

## 📢 Punchy Spoken Script & Cell-by-Cell Walkthrough

### 👤 Speaker 1: Problem Solved, Requirements & Data Ingestion (0:00 — 1:30)

#### 🎬 Stage Cue:
Start at **Cell 0 (Title Slide)**. Scroll smoothly through **Cells 2.1 to 2.2b**, pause explicitly on **Cell 3.0 (Artifact Loader)**, and display **Cells 2.3 to 2.7** highlighting the printed output stats and EDA charts.

#### 🗣️ Word-for-Word Spoken Script:
> "Good morning respected judges and guests. We are Team ML_SE_ALL_02 presenting our applied project: **Early Sepsis Warning with Utility-Based Clinical Thresholds**.
> 
> *(Point to Cell 0)*
> Starting right here at **Cell 0**, the exact problem we are solving is detecting ICU sepsis **hours before clinical onset** while strictly penalizing false alarms to prevent doctor 'Alarm Fatigue.' This requires building an AI that optimizes for clinical utility rather than raw accuracy.
> 
> *(Scroll through Cells 2.1, 2.2, 2.2b, and pause on Cell 3.0)*
> In **Cells 2.1 and 2.2**, we install our specialized stack: `xgboost`, `lightgbm`, and `optuna`. Notice **Cell 2.2b**: our `FORCE_RETRAIN` gate controls execution. Set to `False`, it triggers **Cell 3.0**, our Centralized Artifact Loader, which instantly pre-loads pre-trained models into memory in 60 seconds for live demonstrations.
> 
> *(Scroll to Cells 2.3 and 2.4)*
> To fulfill our data requirements, **Cells 2.3 and 2.4** ingest the PhysioNet 2019 dataset across two hospital systems. Look at the printed output in **Cell 2.4**: we processed exactly **40,336 patient stays** totaling over **1.55 million hourly records**.
> 
> *(Point to EDA Figures in Cells 2.5, 2.6, and 2.7)*
> Our EDA figures in **Cells 2.5 through 2.7** expose the two extreme technical hurdles this project required us to solve: First, the pie chart in **Cell 2.5** shows a severe **1.8% positive class imbalance**. Second, the bar chart in **Cell 2.6** reveals lab tests are missing over 90% of the time. Member 2 will now explain the preprocessing steps we followed to overcome this."

#### 🔬 Speaker 1 Cell-by-Cell Commentary Guide:
* **Cell 0 (Title Block)**: Establishes project title, authors, matriculation numbers, and goal.
* **Cell 2.1 (Dependencies)**: Installs `xgboost`, `lightgbm`, `optuna`, and `joblib`.
* **Cell 2.2 (Imports)**: Imports core Python packages (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
* **Cell 2.2b (Run Control Gate)**: Defines `FORCE_RETRAIN = False` and checks disk for all 8 saved `.joblib` model/prediction artifacts.
* **Cell 3.0 (Centralized Artifact Loader)**: Runs when `should_train = False`. Loads feature columns, models, validation/test predictions, and optimal cutoff into memory in ~60 seconds.
* **Cell 2.3 (`load_all_patients`)**: Ingests `.psv` files, extracts patient IDs, and tags hospital origins (`A` vs `B`).
* **Cell 2.4 (Execution Stats)**: Prints overall ingestion statistics: **40,336 patients**, **1,552,210 rows**, sepsis rate **1.8%**.
* **Cells 2.5–2.7 (EDA Visualizations)**: Generates the label distribution pie chart (`label_distribution.png`), lab missingness bar chart (`missingness.png`), and vital sign trajectories.

---

### 👤 Speaker 2: Steps Followed — Preprocessing & Feature Engineering (1:30 — 3:00)

#### 🎬 Stage Cue:
Reference **Section 3**, scroll through **Cells 4.1 to 4.5**, explain the vectorized math in **Cell 4.6**, and point to the green assertion printout in **Cells 4.8 and 4.9**.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 1. Following our **Section 3 Methodology Roadmap**, we designed a rigorous clinical preprocessing pipeline in **Section 4**.
> 
> *(Scroll through Cells 4.1 to 4.5)*
> In **Cells 4.1 and 4.2**, we define 81 feature columns and tag patient eligibility. To solve missing vital signs, standard mean imputation fails—replacing a collapsed patient's heart rate with the hospital average masks deterioration! Instead, **Cell 4.4** executes **Clinical Forward-Filling**, carrying last-known readings forward until re-tested. Start-of-stay gaps are median-filled in **Cell 4.5**, while **Cell 4.3** creates binary indicators to capture doctor test-ordering frequency.
> 
> *(Point to Cell 4.6)*
> The most critical engineering step occurs in **Cell 4.6**. Doctors diagnose via momentum, so our vectorized `build_windowed_features` engine constructs **6-hour sliding windows**. For every vital sign at hour $T \ge 6$, we extract current value (`_last`), rolling mean (`_mean`), and linear regression velocity (**slope `_slope`**). A rising heart rate slope flags acute deterioration hours before static thresholds break! We also compute 6-hour lab missingness rates (`_miss_rate`).
> 
> *(Point to Cells 4.8 and 4.9)*
> Finally, to prevent time-series data leakage across correlated patient hours, **Cell 4.8** executes a strict **Patient-Level Split** via `GroupShuffleSplit` (70% Train, 10% Val, 20% Test). Look at **Cell 4.9**: our code assertion verifies exactly **0 overlapping patients**. Member 3 will walk through our modeling steps."

#### 🔬 Speaker 2 Cell-by-Cell Commentary Guide:
* **Section 3 (Methodology Roadmap)**: Visual mapping of data transformation and pipeline steps.
* **Cells 4.1–4.2**: Categorizes variables (`VITALS`, `LABS`, `DEMOGRAPHICS`) and classifies patient eligibility (`never_septic`, `early_warning_eligible`, `immediate_only`).
* **Cell 4.3 (Missingness Flags)**: Creates binary indicators (`1` if measured, `0` if NaN) tracking lab order patterns.
* **Cells 4.4–4.5 (Imputation)**: Executes clinical `.groupby('patient_id').ffill()` forward-filling followed by population median filling.
* **Cell 4.6 (`build_windowed_features`)**: Slices 6h windows. Computes `_last`, 6h `_mean`, linear regression `_slope` via `np.polyfit`, and lab `_miss_rate`. Drops hours 1–5.
* **Cell 4.7**: Attaches eligibility tags to the engineered matrix.
* **Cell 4.8 (`patient_level_split`)**: Uses `GroupShuffleSplit` on unique `patient_id`s to partition 70% Train, 10% Val, 20% Test.
* **Cell 4.9 (Leakage Assertion)**: Automated Python `assert` confirming set intersection between train/val/test patient IDs equals `0`.

---

### 👤 Speaker 3: Steps Followed — Modeling & Threshold Calibration (3:00 — 4:30)

#### 🎬 Stage Cue:
Scroll through **Cells 5.1 to 5.5b**, explain **Cell 5.6 (Feature Importances)**, display **Cell 6.1 (Scorer)**, and point directly to the **Threshold Sweep Figure (`threshold_sweep.png`)** in **Cells 6.4 and 6.6**.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 2. In **Section 5**, we execute our machine learning modeling steps.
> 
> *(Scroll through Cells 5.1 to 5.5b)*
> We first evaluated a manual clinical baseline in **Cell 5.1** (`qSOFA`), which achieved a negligible validation utility of `0.005`. Next, we trained three algorithms: regularized ElasticNet Logistic Regression in **Cell 5.2**, gradient boosted XGBoost tuned across 50 Optuna trials in **Cell 5.3**, and LightGBM tuned across 50 trials in **Cell 5.4**. In **Cell 5.3**, we set `scale_pos_weight` to directly penalize missed sepsis alarms. In **Cell 5.5b**, grid search locked in our optimal **Soft-Vote Ensemble blend**: 45% XGBoost, 45% LightGBM, and 10% Logistic Regression. Notice the bar chart in **Cell 5.6**: our engineered 6-hour slopes dominate the top feature importances!
> 
> *(Point to Cell 6.1 and Figure in Cell 6.6)*
> To evaluate these models accurately, **Cell 6.1** embeds the official **PhysioNet Utility Scorer** (+1.0 reward for alarms 12h–6h early, -0.05 penalty for false alarms). This brings us to our final calibration step in **Cell 6.4**. Standard ML triggers at probability 0.50. Under 1.8% class imbalance, models rarely reach 0.50 confidence, causing high miss rates. Look at the threshold sweep graph in **Cell 6.6**: the default 0.50 cutoff yields a poor utility of `0.08`. But sweeping cutoffs revealed our optimal peak at **P ≥ 0.34**, driving validation utility up to **0.341**. Member 4 will now present our final results and live findings."

#### 🔬 Speaker 3 Cell-by-Cell Commentary Guide:
* **Cell 5.1 (qSOFA Baseline)**: Evaluates manual clinical checklist ($+1$ if Resp $> 22$, $+1$ if SBP $< 100$). Validation Utility: `0.005`.
* **Cell 5.2 (Logistic Regression)**: Trains ElasticNet regularized linear model. Saves `lr_model.joblib`.
* **Cell 5.3 (XGBoost + Optuna)**: Executes 50 Bayesian tuning trials optimizing depth, learning rate, child weight. Applies `scale_pos_weight`. Saves `xgb_model.joblib`.
* **Cell 5.4 (LightGBM + Optuna)**: Executes 50 tuning trials for leaf-wise gradient boosting. Saves `lgb_model.joblib`.
* **Cell 5.5b (Soft-Vote Ensemble)**: Grid searches probability weights. Discovers optimal mix: **0.45 XGB + 0.45 LGB + 0.10 LR**. Saves `ensemble_weights.joblib`.
* **Cell 5.6 (Feature Importances Figure)**: Plots Top 15 XGBoost predictors, proving 6h rolling slopes and missingness rates drive predictions.
* **Cells 6.1–6.3 (PhysioNet Scorer)**: Verbatim utility reward logic (+1.0 early warning 12h-6h prior, -0.05 false alarm penalty) and timeline visualization.
* **Cell 6.4 (Threshold Calibration Sweep)**: Sweeps decision cutoffs from `0.05` to `0.95` on Validation data. Proves 0.50 fails (`0.08`) and locks peak cutoff at **P ≥ 0.34** (`0.341`).
* **Cells 6.5–6.6 (Comparison Tables & Curves)**: Displays summary table and overlays validation sweep graphs.

---

### 👤 Speaker 4: Results, Findings & Live Demonstration (4:30 — 6:15)

#### 🎬 Stage Cue:
Point to the **Section 7 Test Table**, walk through the **Cell 8.4 Error Trajectory Figures**, highlight **Cell 9.1 Conclusion Summary**, and execute/stream **Cells 10.1 to 10.5** showing the alarm firing at Hour 18.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 3. Locking our calibrated 0.34 threshold, we evaluated the Ensemble on the completely unseen 20% Test set in **Section 7**.
> 
> *(Point to Section 7 Output Table)*
> Here are our definitive project results: Our calibrated ensemble achieved an official Test Clinical Utility Score of **0.327** and an AUPRC of **0.098**—outperforming the hospital qSOFA baseline by **over 60-fold**! Most importantly, our findings confirm the system gives ICU doctors a median advance warning lead time of **6.0 full hours** before sepsis onset, performing consistently across both Hospital A and Hospital B.
> 
> *(Highlight Figures in Cells 8.1 to 8.4 and Table in Cell 9.1)*
> In **Cells 8.1 through 8.4**, our visual case studies uncovered a critical clinical finding: roughly **25% of septic patients arrive at the hospital already in septic shock on Hour 1**. It is structurally impossible for any monitor to give a 6-hour advance warning for a patient arriving already septic! Being transparent about this explains why the theoretical utility ceiling in ICUs is ~0.65, not 1.0. **Cell 9.1** summarizes these answers neatly.
> 
> *(Execute Live Stream in Cells 10.1 to 10.5)*
> Finally, look at our live stream demonstration in **Cells 10.3 and 10.4**. Watch as we stream an unseen patient's vitals hour-by-hour: everything appears stable until Hour 18, where rising vital sign slopes push our ensemble probability across the 0.34 threshold. The system triggers a live alert **exactly 6 hours before** clinical onset occurs at Hour 24! Thank you judges, we welcome your questions."

#### 🔬 Speaker 4 Cell-by-Cell Commentary Guide:
* **Section 7 (Test Evaluation Table)**: Evaluates ensemble on unseen Test set at `0.34` cutoff. Official Test Utility: **`0.327`** (60x higher than qSOFA). Outputs hospital subgroup comparison.
* **Cells 8.1–8.4 (Error Case Trajectory Figures)**: Visualizes 4 distinct patient trajectories: Success (early alert), Miss (blunted fever), False Alarm (hypertension), and Structural Limit (patient arrives septic on Hour 1).
* **Cell 9.1 (Conclusion Summary Table)**: Summarizes final research answers without recomputing.
* **Cells 10.1–10.5 (Live Demo Walkthrough)**: Loads saved models fresh. Streams an unseen test patient's hourly vital signs. Shows ensemble probability climbing across `0.34` threshold at Hour 18, triggering a live alert **6 hours early**.

---

## 🛡️ Master Team Q&A Coaching Guide

When answering questions, **do not give one-word answers**. Use these crisp explanations:

| Professor's Question | Winning Team Explanation |
|---|---|
| **"Why is your AUPRC score around 0.10? Isn't 0.10 a terrible score?"** | "In balanced datasets yes, but under extreme 1.8% class imbalance, a random guess scores equal to prevalence (0.018). Our score of 0.098 represents more than a **5-fold improvement over random baseline**, and is 5 times higher than qSOFA (0.019)." |
| **"How do we know your Optuna tuning didn't just overfit to validation data?"** | "We prevented overfitting through three layers: 5-fold cross-validation inside Optuna, strict patient-level separation via GroupShuffleSplit, and empirical proof: our Utility score held completely stable on the unseen 20% Test set (**0.327 Test vs 0.341 Validation**)." |
| **"What happens if I change `FORCE_RETRAIN = True` and execute right now?"** | "The centralized loader bypasses disk loading and triggers the 7-minute training pipeline: loading `.psv` files, computing 6h rolling slopes, running 50 Optuna optimization trials per boosted model, blending weights, re-calibrating threshold, and saving new artifacts." |
| **"Why did you include `immediate_only` patients if you know you can't predict them early?"** | "To preserve strict data integrity and clinical transparency. Filtering out patients who arrive already septic would artificially inflate our score to look better than reality. Instead, we report overall results honestly and explain the ~25% structural admission limitation in our error analysis." |
