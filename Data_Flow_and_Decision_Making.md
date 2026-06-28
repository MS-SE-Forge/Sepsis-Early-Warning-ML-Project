# 📊 Data Flow & Decision-Making Architecture

This document maps out the complete end-to-end data pipeline and decision gates for the Early Sepsis Warning project. Use these visual flows to explain your methodology clearly to judges or in technical documentation.

---

## 1. System Execution Pipeline & Run Control Routing

The notebook is architected with a dual-execution path controlled by Cell 10 (`FORCE_RETRAIN`) and automatic file validation. This guarantees live demonstration smoothness while maintaining full technical reproducibility.

```mermaid
graph TD
    A["Raw Dataset Directories<br/>(training_setA / training_setB)"] --> B{"Cell 10: Run Control Gate<br/>Check FORCE_RETRAIN & Files"}
    
    B -->|FORCE_RETRAIN = False<br/>& All Artifacts Present| C["⚡ Fast-Load Mode (~60 seconds)<br/>Load Pre-Trained .joblib & .csv Files"]
    B -->|FORCE_RETRAIN = True<br/>or Any Missing Artifact| D["🔁 Full Training Pipeline (~5–10 minutes)<br/>Execute Complete ML Training"]
    
    subgraph Core ML Training Pipeline
        D --> E["Data Cleaning & Preprocessing<br/>Forward Fill → Median Fill"]
        E --> F["6-Hour Sliding Window Feature Engineering<br/>Extract Vitals (_last, _mean, _slope) & Lab Miss Rates"]
        F --> G["Leakage-Free Patient-Level Split<br/>GroupShuffleSplit: Train (70%) | Val (10%) | Test (20%)"]
        
        G --> H["Model Training & Optuna Tuning<br/>qSOFA Baseline → LR → XGBoost → LightGBM"]
        H --> I["Soft-Vote Ensemble Weight Search<br/>Optimal Grid Search: 45% XGB + 45% LGB + 10% LR"]
        I --> J["Save Output Artifacts to Disk<br/>Overwrite Models & Predictions in artifacts/"]
    end
    
    C --> K["Section 6 & 7: Validation & Evaluation"]
    J --> K
    
    K --> L["Utility Threshold Sweep (Validation Set Only)<br/>Identify Optimal Decision Cutoff: 0.34"]
    L --> M["Final Evaluation on Unseen Test Set<br/>Utility Score + Lead Time + Subgroup Analysis"]
```

---

## 2. Preprocessing & Feature Engineering Flow

How raw hospital measurements transform into actionable predictive features:

```mermaid
flowchart LR
    Raw["Raw .psv Files<br/>(1 row per ICU hour)"] --> Merge["Load & Tag Hospital Source<br/>(Hospital A vs Hospital B)"]
    Merge --> Fill["Clinical Imputation<br/>1. Forward Fill (Last Known Value)<br/>2. Median Fill (Start of Stay NaNs)"]
    Fill --> Window["6-Hour Sliding Window Matrix<br/>Compute Rolling Trends"]
    
    subgraph Engineered Features per Hour T
        Window --> Vitals["Vital Trends<br/>• _last (Current Value)<br/>• _mean (6h Average)<br/>• _slope (Linear Trend)"]
        Window --> Labs["Lab Signals<br/>• _last (Last Measured)<br/>• _miss_rate (Test Order Rate)"]
    end
```

### Key Preprocessing Decisions
* **Why Forward-Fill?** In an ICU setting, physiological variables do not change instantly. Carrying the last measured heart rate or blood pressure forward reflects the true clinical assumption until a new measurement is taken.
* **Why Missingness Rates?** Lab tests (like Lactate or Blood Cultures) are not ordered randomly. A sudden cluster of lab orders indicates doctor concern. By calculating the 6-hour missingness rate (`_miss_rate`), the AI learns to treat the *presence or absence of a test* as a diagnostic signal.
* **Why Slopes?** A patient with a static Heart Rate of 95 bpm is stable. A patient whose Heart Rate rose from 70 to 95 over the last 6 hours (`positive slope`) is actively deteriorating. Slopes capture momentum before static thresholds break.

---

## 3. Clinical Utility Decision Matrix (PhysioNet Challenge 2019)

Standard Machine Learning metrics (Accuracy, F1-Score) treat all errors equally. In medical monitoring, a late warning is fatal, and excessive false alarms cause doctor burnout. We evaluate models using the official clinical utility reward function:

```mermaid
graph LR
    subgraph Timeline of a Sepsis Patient Stay
        Admission["ICU Admission<br/>(t = 0)"] --> Optimal["Optimal Warning Window<br/>(12h to 6h before onset)<br/>Reward: +1.0 Maximum Utility"]
        Optimal --> Late["Late Warning Window<br/>(Within 6h of onset or after)<br/>Reward: Drops steadily to 0.0"]
        Late --> Onset["Clinical Sepsis Onset<br/>(t_sepsis)"]
    end
```

### The Reward / Penalty Rulebook
| Clinical Scenario | System Action | Utility Reward / Penalty | Interpretation |
|---|---|---|---|
| **Early True Positive** | Alarm sounds **6 to 12 hours before** sepsis onset | **+1.00** (Maximum Reward) | Provides doctors enough lead time to administer fluids and intravenous antibiotics safely. |
| **Late True Positive** | Alarm sounds **within 6 hours** of onset or after | **+0.80 down to 0.00** | Better than nothing, but clinical efficacy drops rapidly as organ damage begins. |
| **False Negative (Miss)**| Patient develops sepsis, but **no alarm** sounds | **0.00** (No Reward) | Failed to protect the patient. |
| **False Positive (False Alarm)**| Patient never develops sepsis, but **alarm sounds** | **-0.05** (Active Penalty) | Contributes to "alarm fatigue," causing staff to ignore future monitor alerts. |
| **True Negative** | Stable patient, **no alarm** sounds | **0.00** (Neutral) | Correct baseline system behavior. |

---

## 4. Threshold Calibration Gate

Why we do not use the default Machine Learning probability cutoff of `0.50`:

```mermaid
graph TD
    Prob["Ensemble Raw Probability<br/>P(Sepsis) ∈ [0.0, 1.0]"] --> Sweep{"Threshold Sweep on Validation Set<br/>Test cutoffs from 0.05 to 0.95"}
    
    Sweep -->|Default Cutoff: 0.50| Fail["❌ Extreme Class Imbalance (1.8% Pos)<br/>Model rarely reaches 0.50 probability<br/>Result: High Miss Rate (Utility ≈ 0.08)"]
    
    Sweep -->|Optimal Cutoff: 0.34| Win["✅ Calibrated Clinical Utility<br/>Captures deteriorating patients early<br/>Result: Maximized Utility Score (0.327)"]
```
