# 📊 Data Flow & Decision-Making Architecture (Visual Cell-Annotated)

This document maps out the complete end-to-end data pipeline and decision gates for the Early Sepsis Warning project. Every single architectural block across all four diagrams is explicitly annotated with its **exact visual notebook heading** (e.g., **Cell 2.2b**, **Cell 4.6**, **Cell 5.3**, **Cell 6.4**) following your notebook's exact structure.

---

## 1. System Execution Pipeline & Visual Cell Routing

The notebook is architected with a dual-execution path controlled by **Cell 2.2b (`FORCE_RETRAIN`)**. This guarantees instant presentation execution (~60 seconds) while preserving full technical training capability (~7 minutes).

```mermaid
graph TD
    A["📂 Raw Dataset Directories<br/>Cell 2.3: Auto-detect paths - training_setA / setB<br/>Cell 2.4: Load 40,336 patients - ~1.5M hourly rows"] --> B{"⚖️ Cell 2.2b: Run Control Gate<br/>Check FORCE_RETRAIN and Artifact Existence"}
    
    B -->|FORCE_RETRAIN = False<br/>and All Files Present| C["⚡ Fast-Load Mode ~60 seconds<br/>Cell 3.0 Centralized Loader: imports pre-trained<br/>.joblib models and predictions.csv instantly"]
    B -->|FORCE_RETRAIN = True<br/>or Missing Files| D["🔁 Full Training Pipeline ~7 minutes<br/>Execute active machine learning training"]
    
    subgraph Core Preprocessing and Feature Extraction Pipeline
        D --> E["🧹 Preprocessing and Imputation<br/>Cell 4.2: Tag patient eligibility groups<br/>Cell 4.3: Create binary lab missingness flags<br/>Cell 4.4: Forward-fill vital history per patient<br/>Cell 4.5: Median fill start-of-stay NaNs"]
        E --> F["⏱️ 6-Hour Sliding Window Engineering<br/>Cell 4.6: build_windowed_features computes rolling<br/>means, linear regression slopes and lab miss rates"]
        F --> G["🔒 Leakage-Free Patient-Level Split<br/>Cell 4.8: GroupShuffleSplit - 70% Train | 10% Val | 20% Test<br/>Cell 4.9: Assert 0 overlapping patients across splits"]
    end
    
    subgraph Model Training and Tuning Pipeline
        G --> H["🤖 Model Training and Hyperparameter Tuning<br/>Cell 5.1: Evaluate qSOFA clinical baseline<br/>Cell 5.2: Train ElasticNet Logistic Regression<br/>Cell 5.3: XGBoost training + 50 Optuna tuning trials<br/>Cell 5.4: LightGBM training + 50 Optuna tuning trials"]
        H --> I["🤝 Soft-Vote Ensemble Optimization<br/>Cell 5.5b: Grid search best probability weights<br/>Discovers: 45% XGB + 45% LGB + 10% LR"]
        I --> J["💾 Save Output Artifacts to Disk<br/>Cell 5.5b: Export models and CSVs to artifacts/ dir"]
    end
    
    C --> K["📈 Evaluation and Threshold Calibration"]
    J --> K
    
    subgraph Evaluation and Calibration Pipeline
        K --> L["📐 PhysioNet Utility Scorer and Threshold Sweep<br/>Cell 6.1: Define official verbatim reward function<br/>Cell 6.2: Load validation and test prediction curves<br/>Cell 6.4: Validation Sweep tests cutoffs 0.05 to 0.95<br/>Discovers Optimal Clinical Threshold: P ≥ 0.34"]
        L --> M["🏆 Test Set and Subgroup Verification<br/>Section 7: Compute final Test Utility - 0.327<br/>Stratify results across Hospital A vs Hospital B"]
        M --> N["🔬 Error Analysis and Live Code Demo<br/>Cell 8.4: Deep-dive 4 clinical case study plots<br/>Cell 10.3-10.5: Live streaming inference demo"]
    end
```

---

## 2. Preprocessing & Feature Engineering Flow (Annotated)

How raw hospital measurements transform into actionable predictive features inside **Section 4**:

```mermaid
flowchart LR
    Raw["📂 Raw .psv Files<br/>Cell 2.3 and 2.4<br/>1 row per ICU hour"] --> Merge["🏥 Load and Tag Hospital Source<br/>Cell 2.4<br/>Hospital A vs Hospital B"]
    Merge --> Fill["🧹 Clinical Imputation<br/>Cell 4.4: Forward Fill - Last Known Value<br/>Cell 4.5: Median Fill - Start of Stay NaNs"]
    Fill --> Window["⏱️ 6-Hour Sliding Window Matrix<br/>Cell 4.6: build_windowed_features"]
    
    subgraph Engineered Features per Hour T - Cell 4.6
        Window --> Vitals["📈 Vital Trends<br/>• _last - Current Value<br/>• _mean - 6h Average<br/>• _slope - Linear Trend"]
        Window --> Labs["🧪 Lab Signals<br/>• _last - Last Measured<br/>• _miss_rate - Test Order Rate"]
    end
```

### Key Preprocessing Decisions
* **Why Forward-Fill (Cell 4.4)?** In an ICU setting, physiological variables do not change instantly. Carrying the last measured heart rate or blood pressure forward reflects the true clinical assumption until a new measurement is taken.
* **Why Missingness Rates (Cell 4.3 & 4.6)?** Lab tests (like Lactate or Blood Cultures) are not ordered randomly. A sudden cluster of lab orders indicates doctor concern. By calculating the 6-hour missingness rate (`_miss_rate`), the AI learns to treat the *presence or absence of a test* as a diagnostic signal.
* **Why Slopes (Cell 4.6)?** A patient with a static Heart Rate of 95 bpm is stable. A patient whose Heart Rate rose from 70 to 95 over the last 6 hours (`positive slope`) is actively deteriorating. Slopes capture momentum before static thresholds break.

---

## 3. Clinical Utility Decision Matrix (Cell 6.1 & 6.3)

Standard Machine Learning metrics (Accuracy, F1-Score) treat all errors equally. In medical monitoring, a late warning is fatal, and excessive false alarms cause doctor burnout. We evaluate models using the official clinical utility reward function embedded in **Cell 6.1**:

```mermaid
graph LR
    subgraph Timeline of a Sepsis Patient Stay - Cell 6.3 Visualization
        Admission["🏥 ICU Admission<br/>t = 0"] --> Optimal["🟢 Optimal Warning Window<br/>12h to 6h before onset<br/>Reward: +1.0 Maximum Utility"]
        Optimal --> Late["🟡 Late Warning Window<br/>Within 6h of onset or after<br/>Reward: Drops steadily to 0.0"]
        Late --> Onset["🔴 Clinical Sepsis Onset<br/>t_sepsis"]
    end
```

### The Reward / Penalty Rulebook (Cell 6.1 Logic)
| Clinical Scenario | System Action | Utility Reward / Penalty | Interpretation |
|---|---|---|---|
| **Early True Positive** | Alarm sounds **6 to 12 hours before** sepsis onset | **+1.00** (Maximum Reward) | Provides doctors enough lead time to administer fluids and intravenous antibiotics safely. |
| **Late True Positive** | Alarm sounds **within 6 hours** of onset or after | **+0.80 down to 0.00** | Better than nothing, but clinical efficacy drops rapidly as organ damage begins. |
| **False Negative (Miss)**| Patient develops sepsis, but **no alarm** sounds | **0.00** (No Reward) | Failed to protect the patient. |
| **False Positive (False Alarm)**| Patient never develops sepsis, but **alarm sounds** | **-0.05** (Active Penalty) | Contributes to "alarm fatigue," causing staff to ignore future monitor alerts. |
| **True Negative** | Stable patient, **no alarm** sounds | **0.00** (Neutral) | Correct baseline system behavior. |

---

## 4. Threshold Calibration Gate (Cell 6.4)

Why we do not use the default Machine Learning probability cutoff of `0.50` in **Cell 6.4**:

```mermaid
graph TD
    Prob["📊 Ensemble Raw Probability<br/>Cell 5.5b Output: P Sepsis in range 0.0 to 1.0"] --> Sweep{"⚖️ Cell 6.4: Threshold Sweep on Validation Set<br/>Test cutoffs systematically from 0.05 to 0.95"}
    
    Sweep -->|Default Cutoff: 0.50| Fail["❌ Extreme Class Imbalance - 1.8% Pos<br/>Model rarely reaches 0.50 probability<br/>Result: High Miss Rate - Utility ≈ 0.08"]
    
    Sweep -->|Optimal Cutoff: 0.34| Win["✅ Calibrated Clinical Utility<br/>Captures deteriorating patients early<br/>Result: Maximized Utility Score - 0.341 Val / 0.327 Test"]
```

---

## 5. Visual Cell-by-Cell Detailed Execution Summary

Use this exact mapping to match your presentation points directly to the visual markdown headers in your notebook:

### ⚙️ Section 2: Setup and Exploratory Data Analysis (EDA)
* **Title Slide (Section 0)**: Displays project metadata, team members, matriculation numbers, and deadline.
* **Cell 2.1 (Dependencies)**: Executes a single quiet `pip install` for `xgboost`, `lightgbm`, `optuna`, `joblib`, etc.
* **Cell 2.2 (Imports)**: Imports essential Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `os`, `json`).
* **Cell 2.2b (Run Control Gate)**: Defines `FORCE_RETRAIN = False`. Checks if all 8 artifact files exist in `artifacts/`. Sets `should_train`.
* **Centralized Artifact Loader (Cell 3.0)**: If `should_train = False`, pre-loads feature columns, models, validation/test predictions, and optimal threshold into memory instantly.
* **Cell 2.3 (`load_all_patients`)**: Scans directories for `.psv` files, extracts patient IDs from file names, attaches hospital tags (`A` or `B`), and concatenates them into one DataFrame.
* **Cell 2.4**: Calls `load_all_patients()`. Prints summary stats: **40,336 patients**, **1,552,210 rows**, overall sepsis rate **1.8%**.
* **Cells 2.5, 2.6, 2.7 (EDA Plots)**: Generates the label distribution pie chart (`label_distribution.png`), the lab missingness bar chart (`missingness.png`), and sample vital trajectory plots.

### 🧹 Section 4: Clinical Preprocessing & Feature Engineering
* **Cell 4.1**: Lists feature arrays: `VITALS` (8 columns), `LABS` (26 columns), `DEMOGRAPHICS` (6 columns).
* **Cell 4.2 (Eligibility Tagging)**: Classifies patients into `never_septic`, `early_warning_eligible` (sepsis onset ≥ hour 7), and `immediate_only` (sepsis onset within first 6 hours).
* **Cell 4.3 (Missingness Indicators)**: Creates binary flags (`1` if lab measured, `0` if NaN) to capture doctor test-ordering frequency.
* **Cell 4.4 (Forward Fill)**: Carries each patient's last measured vital sign forward hour-by-hour.
* **Cell 4.5 (Median Fill)**: Imputes remaining start-of-stay NaNs using population median.
* **Cell 4.6 (`build_windowed_features`)**: For every hour $T \ge 6$, slices the past 6 hours. Computes `_last`, rolling 6h `_mean`, linear fit `_slope`, and lab `_miss_rate`. Drops hours 1–5 per patient.
* **Cell 4.7**: Attaches eligibility group labels to the processed feature table.
* **Cell 4.8 (`patient_level_split`)**: Uses `GroupShuffleSplit` on unique `patient_id`s to divide data into **Train (70%)**, **Validation (10%)**, and **Test (20%)**.
* **Cell 4.9 (Leakage Assertion)**: Runs automated Python `assert` verifying intersections between train/val/test patient sets equal `0`.

### 🤖 Section 5 & 6: Modeling and Threshold Calibration
* **Cell 5.1 (qSOFA Baseline)**: Computes clinical baseline score ($+1$ if Resp $> 22$, $+1$ if SBP $< 100$). Validation Utility: `0.005`.
* **Cell 5.2 (Logistic Regression)**: Trains ElasticNet regularized model. Saves `lr_model.joblib`.
* **Cell 5.3 (XGBoost + Optuna)**: Optimizes tree depth, learning rate, and child weight over 50 trials using `scale_pos_weight` to handle imbalance. Saves `xgb_model.joblib`.
* **Cell 5.4 (LightGBM + Optuna)**: Runs 50 tuning trials for LightGBM. Saves `lgb_model.joblib`.
* **Cell 5.5b (Soft-Vote Ensemble)**: Performs grid search over probability weights. Discovers optimal mix: **0.45 XGB + 0.45 LGB + 0.10 LR**. Saves `ensemble_weights.joblib`.
* **Cell 5.6**: Plots Top 15 Feature Importances from XGBoost.
* **Cell 6.1 (PhysioNet Utility Scorer)**: Verbatim official function awarding $+1.0$ for alarms 12h–6h prior, grading down to $0.0$ for late alarms, and penalizing false alarms $-0.05$.
* **Cell 6.2**: Loads saved predictions and ensemble weights in fast-load mode.
* **Cell 6.3**: Plots the official PhysioNet reward/penalty timeline curve.
* **Cell 6.4 (Threshold Sweep)**: Evaluates decision cutoffs from `0.05` to `0.95` on Validation set. Proves default `0.50` fails (Utility `0.08`) and discovers peak clinical cutoff at **`0.34`** (Utility `0.341`).
* **Cells 6.5 & 6.6**: Displays per-model utility comparison tables and overlays validation sweep curves.

### 🏆 Section 7, 8, 9 & 10: Evaluation, Error Analysis & Live Demo
* **Section 7 (Test Evaluation)**: Evaluates ensemble on unseen Test set at `0.34` cutoff. Official Test Utility: **`0.327`** (60x higher than qSOFA). Outputs hospital subgroup table.
* **Cells 8.1–8.4 (Error Case Studies)**: Plots 4 distinct patient trajectories: Success (early alert), Miss (blunted fever), False Alarm (chronic hypertension), and Structural Limit (patient arrives septic on Hour 1).
* **Cell 9.1 (Conclusion Table)**: Summarizes primary findings answering research question without recomputing.
* **Cells 10.1–10.5 (Live Demo)**: Loads saved models fresh. Streams an unseen test patient's hourly vital signs. Shows ensemble probability climbing across `0.34` threshold at Hour 18, triggering a live alert **6 hours early**.

---

## 6. Project Dependencies & Technical Roles

Installed via **Cell 2.1** and imported in **Cell 2.2**, each package plays a vital, specialized role in executing the clinical pipeline:

| Library / Package | Version | Technical Role & Usage in Notebook |
|---|---|---|
| **`pandas`** | Core | Data manipulation, tabular slicing, groupby operations, forward/median clinical imputation (**Cell 4.4, 4.5**), and vectorized 6-hour rolling aggregations (**Cell 4.6**). |
| **`numpy`** | Core | High-performance numerical arrays, missingness calculations, and linear polynomial regression (`np.polyfit`) to extract vital trajectory deterioration slopes (**Cell 4.6**). |
| **`scikit-learn`** | Core | Provides `GroupShuffleSplit` for leakage-free patient separation (**Cell 4.8**), `LogisticRegression` with ElasticNet regularization (**Cell 5.2**), and evaluation metrics (`roc_auc_score`, `average_precision_score`). |
| **`xgboost`** | Latest | Extreme Gradient Boosting tree algorithm (**Cell 5.3**). Models complex non-linear vital interactions and tackles extreme 1.8% class imbalance using the `scale_pos_weight` hyperparameter. |
| **`lightgbm`** | Latest | Light Gradient Boosting Machine (**Cell 5.4**). Histogram-based, leaf-wise tree growth algorithm that complements XGBoost and excels at recognizing sparse lab missingness patterns. |
| **`optuna`** | Latest | Automated hyperparameter tuning framework (**Cell 5.3, 5.4**). Uses Tree-structured Parzen Estimators (Bayesian optimization) across 50 trials to tune tree depth, learning rates, and L1/L2 regularization. |
| **`joblib`** | Core | Object serialization library (**Cell 3.0, 5.5b**). Saves and loads trained `.joblib` models and ensemble weights instantaneously, powering the 60-second Fast-Load presentation mode. |
| **`matplotlib` & `seaborn`** | Core | Clinical visualization engine. Generates exploratory charts (**Cell 2.5–2.7**), threshold calibration sweep curves (**Cell 6.4**), precision-recall graphs, and the 4 clinical error case studies (**Cell 8.4**). |
