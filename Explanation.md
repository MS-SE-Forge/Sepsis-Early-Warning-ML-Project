# 🏥 Early Sepsis Warning — Notebook README

**Project:** Early Sepsis Warning with Utility-Based Clinical Thresholds  
**Course:** Machine Learning — Summer 2026  
**Team:** ML_SE_ALL_02

---

## 🌟 What Is This Notebook About?

Imagine you are a doctor watching over many sick patients in an ICU (Intensive Care Unit). Every hour, a machine records each patient's vital signs — things like heart rate, blood pressure, and breathing speed. Your job is to catch **sepsis** before it becomes life-threatening.

Sepsis is like a fire inside the body. The immune system, which normally fights infections, goes out of control and starts attacking the body's own organs. Every hour of delay costs lives — roughly **7% more people die for each hour treatment is delayed**.

This notebook builds a **sepsis alarm system**. It looks at each patient's hourly readings and asks: *"Is this person likely to develop sepsis in the next few hours?"* If yes, it fires an alarm — early enough for doctors to act.

The tricky part is this: we don't just want to be *accurate*. We need to be **useful**. Shouting "FIRE!" every five minutes trains doctors to ignore the alarm (called "alarm fatigue"). So the notebook carefully balances **catching real sepsis early** against **not causing too many false alarms**.

---

## 🗂️ How Is the Notebook Organised?

The notebook follows a clean pipeline:

```
Load Data → Clean & Prepare → Train Models → Pick Best Alarm Level → Evaluate → Report
```

Here is what each major section does:

| Section | What It Does |
|---|---|
| Section 1 | Explains the problem in plain language |
| Section 2 | Loads the patient dataset |
| Section 3 | Describes the plan (methodology) |
| Section 4 | Cleans and prepares the data |
| Section 5 | Trains four different alarm models |
| Section 6 | Picks the best alarm threshold using the official scoring system |
| Section 7 | Tests everything on patients the model has never seen |
| Section 8 | Studies cases where the model succeeded and failed |
| Section 9 | Summarises what was found |
| Section 10 | Live demonstration on real patients |

---

## 📦 Section 2 — Setup

### Cell 2.1 — Installing Packages

```python
subprocess.check_call([sys.executable, '-m', 'pip', 'install', ...])
```

**What it does:** Before we can cook a meal, we need to buy the ingredients. This cell installs all the software tools the notebook needs — things like XGBoost (a powerful model), Optuna (a tool that automatically finds the best settings), and Matplotlib (a drawing tool for charts).

**Output:** `✅ All dependencies installed.`

---

### Cell 2.2 — Imports

```python
import numpy as np
import pandas as pd
import xgboost as xgb
...
```

**What it does:** This cell loads all the tools into memory so the rest of the notebook can use them. Think of it like opening all your toolboxes and laying the tools out on the workbench before you start working.

It also sets a **random seed** (`np.random.seed(42)`). This is like shuffling a deck of cards and then marking exactly where each card is — so that every time you "shuffle" again, you get the exact same result. This makes the notebook fully reproducible.

Finally, it creates an `artifacts/` folder. This is where all saved models, charts, and results will be stored.

---

### Cell 2.2b — The On/Off Switch (`FORCE_RETRAIN`)

```python
FORCE_RETRAIN = False
```

**What it does:** Training the models takes 5–10 minutes. Once trained, the results are saved in the `artifacts/` folder. This cell checks whether those saved files already exist.

- If `FORCE_RETRAIN = False` and the files exist → **fast mode**: load everything instantly (about 60 seconds for the whole notebook).
- If `FORCE_RETRAIN = True` or files are missing → **train from scratch**.

Think of it like this: you baked a cake yesterday and stored it in the fridge. Today you can either eat the stored cake (fast) or bake a brand new one (slow but fresh).

**Output example:** `⚡ Fast-load mode — all 8 artifacts present`

---

## 📂 Section 2 — Loading the Data

### Cell 2.3 — Finding the Dataset

```python
for candidate in ['training_setA', 'training_setB', ...]:
    if os.path.isdir(candidate): DATA_DIRS.append(candidate)
```

**What it does:** The dataset comes as two folders — one for Hospital A and one for Hospital B. Each folder contains thousands of small files, one per patient. This cell automatically searches common folder locations so you don't have to hardcode a path.

**Why two hospitals?** Because a model that only works at one hospital is not very useful in the real world. We want to see if it generalises across different hospitals with different equipment and patient populations.

---

### Cell 2.4 — Counting Patients and Printing Statistics

```python
raw_df = load_all_patients(DATA_DIRS)
```

**What it does:** This cell reads every patient file, sticks them all together into one giant table, and prints a summary.

**Output example:**
```
Total patients       : 40,336
Total patient-hours  : 1,552,210
Septic patients      : 2,932 (7.3%)
Septic rows          : 27,010 (1.74%)
```

The key line at the bottom is a warning: `'Always predict 0' = 98.26% accuracy, 0% clinical value.` This means if we built a model that just said "no sepsis ever", it would be right 98% of the time — but it would miss every real case. That's why raw accuracy is a useless metric here.

---

### Cell 2.5 — 📊 Figure: Label Distribution

```python
axes[0].bar(['Never Septic (0)', 'Septic (1)'], ...)
```

**What it does:** Draws two bar charts — one at the patient level, one at the row level.

**🖼️ What the figure shows:**

Imagine you have a bag of 100 marbles. About 93 are blue (healthy patients) and only 7 are red (septic patients). The left chart shows this imbalance at the patient level. The right chart shows it at the hourly reading level — and there it's even worse, with barely 1.74% of readings being "septic".

This picture is the entire reason why we can't just measure "accuracy" — the blue pile is so much bigger that even a dumb model that ignores red wins on accuracy.

---

### Cell 2.6 — 📊 Figure: Lab Missingness

```python
missing_pct = raw_df.isnull().mean().sort_values(ascending=False) * 100
```

**What it does:** Calculates how often each measurement is missing and draws a horizontal bar chart.

**🖼️ What the figure shows:**

Most vital signs (heart rate, blood pressure) are recorded every hour. But lab tests — like checking someone's Lactate level (a key sepsis marker) — are only done when a doctor orders them. Some labs are missing over 99% of the time.

Think of it like a class register: every student is marked "present" or "absent" every day (vitals), but the teacher only gives a spelling test occasionally (labs). Most of the time, there's no spelling test score at all.

The red dashed line marks 90% missing. The purple dashed line marks 50% missing. Nearly all lab columns are to the right of the 90% line — meaning they are usually blank.

**Why this matters:** A missing lab result isn't just "no data". In a hospital, if a doctor didn't order a test, it might mean they weren't worried — or it might mean they forgot. Either way, "test not done" is itself a useful signal, which is why the next preprocessing step adds missingness flags.

---

### Cell 2.7 — 📊 Figure: Sample Patient Trajectories

```python
plot_patient_trajectory(raw_df, s_pid, ...)  # septic patient
plot_patient_trajectory(raw_df, h_pid, ...)  # healthy patient
```

**What it does:** Plots the heart rate and respiratory rate over time for one septic patient and one healthy patient side-by-side.

**🖼️ What the figure shows:**

There are 4 small plots arranged in 2 rows and 2 columns:
- Left column: the septic patient's HR (top) and Resp rate (bottom). A vertical purple dashed line marks the moment sepsis was officially diagnosed.
- Right column: the healthy patient's HR (top) and Resp rate (bottom). No vertical line (no sepsis).

You can often see the septic patient's readings climbing or becoming more erratic before the purple line. This is the visual version of what the model is trying to learn: *spotting the climb before it becomes a crisis.*

---

## 🏗️ Section 4 — Preprocessing (Cleaning and Preparing the Data)

### Cell 4.1 — Defining Feature Groups

```python
VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', ...]
LABS   = ['Lactate', 'Creatinine', 'WBC', ...]
WINDOW = 6
```

**What it does:** Names and groups all the columns the models will use. Vitals (8 columns) are measured every hour. Labs (26 columns) are drawn occasionally. The `WINDOW = 6` means we look back 6 hours to build trend features.

---

### Cell 4.2 — Eligibility Tagging

```python
def tag_patient_eligibility(df, window_hours=6):
    ...
    group = ('early_warning_eligible' if onset_idx >= window_hours else 'immediate_only')
```

**What it does:** Labels every patient with one of three groups:

| Group | Meaning |
|---|---|
| `never_septic` | Healthy the whole stay |
| `early_warning_eligible` | Got sepsis but at least 6 hours into their ICU stay — we *could* have warned early |
| `immediate_only` | Sepsis hit within the first 6 hours — **impossible** to warn early no matter how good the model |

This is one of the most honest and important steps in the whole notebook. About 25% of septic patients are `immediate_only`. If we mix them into the results, the model looks worse than it really is — because we're penalising it for failing to do something *physically impossible*.

---

### Cell 4.3 — Adding Missingness Flags

```python
df[f'{col}_measured'] = df[col].notna().astype(int)
```

**What it does:** For every lab column, adds a new column that is `1` when the lab was actually drawn that hour, and `0` when it wasn't.

**Why?** Remember: in a hospital, ordering a test is a deliberate clinical decision. A patient who has Lactate drawn three times in one day is being watched more closely than one who has never had it drawn. This "test ordering pattern" carries information about how worried the medical team is — and that information can help predict sepsis.

---

### Cell 4.4 — Forward Fill

```python
df[cols] = df.groupby('patient_id')[cols].ffill()
```

**What it does:** Carries the last known value forward in time. If your Creatinine was measured at hour 5 and not again until hour 11, the model fills hours 6–10 with the hour-5 reading.

Think of it like checking the temperature of your room. You measured it at 8am and it was 22°C. At 9am, 10am, 11am you didn't check — but 22°C is a better guess than "unknown".

---

### Cell 4.5 — Median Fill

```python
df[cols] = df[cols].fillna(df[cols].median())
```

**What it does:** After forward fill, some patients still have missing values at the very beginning of their stay (nothing to carry forward yet). This fills those remaining blanks with the average value across all patients.

It's like a new student joining school. We don't know their height, so we guess the class average until we get their actual measurement.

---

### Cell 4.6 — Building 6-Hour Sliding Window Features

```python
feat_dict[f'{v}_last']  = last    # most recent value
feat_dict[f'{v}_mean']  = mean    # average over last 6 hours
feat_dict[f'{v}_slope'] = slope   # is it going up or down?
```

**What it does:** This is the most important preprocessing step. For each patient-hour, instead of just using the current reading, we summarise the **last 6 hours** into three features per vital sign:

- **`_last`:** The most recent value (what is the heart rate right now?)
- **`_mean`:** The 6-hour average (what has the heart rate been on average?)
- **`_slope`:** Is the reading going up or down? (calculated by fitting a line through the last 6 readings)

For labs: we use the last known value and the fraction of hours in the window where the test was actually drawn.

**Why the slope matters:** Sepsis often shows up as a *trend* — a gradual climb in heart rate and breathing rate. A patient with HR = 100 that has been falling from 120 is probably fine. A patient with HR = 100 that has been climbing from 80 might be in trouble. The slope captures this trend.

---

### Cell 4.7 — Attaching Eligibility Groups

```python
feat_df = feat_df.merge(eligibility_df[['patient_id', 'eligibility_group']], ...)
```

**What it does:** Joins the eligibility labels (from Cell 4.2) onto the main feature table so we always know which group each patient-hour belongs to.

---

### Cell 4.8 — Splitting Patients into Train / Validation / Test

```python
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, ...)
```

**What it does:** Divides all patients into three groups:
- **Training set (70%):** The patients the model learns from.
- **Validation set (10%):** Used to tune settings while training — the model can't "cheat" from this.
- **Test set (20%):** Locked away until the very end. Used only once to report final results.

**The critical detail — `GroupShuffleSplit`:** This ensures that **all rows from one patient stay in the same split**. A patient's hourly readings are closely related to each other. If we randomly split rows, the model might see hour 5 of patient X in training and hour 6 of patient X in testing — and it would just memorise the patient rather than learning general patterns. That's called **data leakage** and it makes results look artificially good.

---

### Cell 4.9 — The Leakage Check (Critical!)

```python
assert len(tr_p & va_p) == 0, '❌ LEAKAGE: train ∩ val!'
assert len(tr_p & te_p) == 0, '❌ LEAKAGE: train ∩ test!'
assert len(va_p & te_p) == 0, '❌ LEAKAGE: val ∩ test!'
```

**What it does:** Triple-checks that no patient appears in more than one split. If any patient did overlap, the code immediately crashes with a clear error message.

This is like an exam invigilator checking that no student has a copy of the exam paper before the test starts. It's a safety guarantee — not just a politeness.

---

## 🤖 Section 5 — Training the Models

### Cell 5.1 — qSOFA Rule (Zero Training)

```python
score = (df['Resp_last'] > 22).astype(int) + (df['SBP_last'] < 100).astype(int)
alarm = (score >= 1)
```

**What it does:** This is the **clinical baseline** — the rule doctors actually use at the bedside today. It adds one point for each of:
- Breathing rate above 22 breaths per minute (tachypnoea — breathing too fast)
- Systolic blood pressure below 100 mmHg (hypotension — too low pressure)

If the score is 1 or more, the alarm fires.

**Why include this?** If our fancy machine learning models can't beat a two-line rule that requires no computer, they have no clinical value. This sets the performance floor everything else must clear.

**Note:** The real qSOFA also checks for confusion (GCS score), but that measurement isn't in this dataset — so the rule here is slightly weaker than the clinical version.

---

### Cell 5.2 — Logistic Regression with ElasticNet

```python
for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    m = LogisticRegression(penalty='elasticnet', ...)
```

**What it does:** Logistic Regression is the simplest possible "learned" model. It draws a straight boundary line through the feature space and says "everything on this side = alarm, everything on the other side = no alarm".

The code tries 7 different values of `C` (a setting that controls how strict or relaxed the model is). It picks the one that scores best on the validation set.

`class_weight='balanced'` tells the model to pay extra attention to the rare sepsis cases — otherwise, it would mostly try to predict "no sepsis" because that's correct 98% of the time.

**Why include it?** It answers the question: "Does any learned model beat the dumb rule?" If even the simplest model beats qSOFA, it validates using ML at all.

---

### Cell 5.3 — XGBoost with Optuna Tuning

```python
def xgb_objective(trial):
    params = dict(
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1),
        max_depth     = trial.suggest_int('max_depth', 4, 8),
        ...
    )
```

**What it does:** XGBoost is a powerful ensemble model that builds many decision trees, each one trying to fix the mistakes of the one before it. (Think of it as a team of 1,000 consultants, each reviewing the previous one's work and improving on it.)

**Optuna** automatically searches for the best settings by running 50 "trials" — 50 different combinations of hyperparameters — and returning the one with the best validation score.

`scale_pos_weight ≈ 55` is a key setting. Since there are roughly 55 healthy rows for every 1 septic row in the training set, this tells XGBoost to treat each septic row as if it were 55 rows. This prevents the model from ignoring the rare class.

---

### Cell 5.4 — LightGBM with Optuna Tuning

```python
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
```

**What it does:** LightGBM does essentially the same job as XGBoost but uses a different internal algorithm that is 3–5× faster to train. It also has `num_leaves` as a setting (instead of `max_depth`) which gives it more flexibility.

We train it the same way — 50 Optuna trials, `scale_pos_weight ≈ 55`, early stopping on the validation AUPRC.

---

### Cell 5.5b — 📊 Ensemble Model + Figure: PR Curve Comparison

```python
ens_probs = xgb_w * xgb_val_probs + lgb_w * lgb_val_probs + lr_w * lr_val_probs
```

**What it does:** Combines all three models into one by taking a **weighted average** of their probability scores:
- XGBoost: 45%
- LightGBM: 45%
- Logistic Regression: 10%

The exact weights are found by trying 16 different weight combinations on the validation set and picking the best one. Mixing models together often works better than any single model alone — like asking several doctors for their opinion and averaging them.

**🖼️ What the figure shows (PR Curve Comparison):**

This chart has **Recall** (how many septic cases did we catch?) on the x-axis and **Precision** (of everything we flagged, how many were actually septic?) on the y-axis.

The four coloured lines represent the four models. The horizontal dashed orange line at the very bottom is the **random baseline** — what a model that just randomly guesses would achieve (about 1.74%, the actual rate of sepsis in the data).

A good model appears as a **high curve** — high up and to the right. The Ensemble (purple) typically sits above the others, meaning it finds more septic cases at a given false alarm rate.

The number in each label, e.g. `(AUPRC=0.0800)`, is the area under the curve. Any value well above 0.0174 means the model is genuinely learning something useful.

---

### Cell 5.6 — 📊 Figure: Feature Importance (Top 15)

```python
importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS)
```

**What it does:** Asks XGBoost: "Which features did you find most useful when making decisions?"

**🖼️ What the figure shows:**

A horizontal bar chart showing the 15 most useful features, sorted from most to least important. Longer bars = more important.

Features like `Resp_last`, `HR_slope`, `HR_mean`, and `SBP_last` typically appear at the top. This makes clinical sense: rising breathing rate and heart rate trends are classic early signs of sepsis. The model learned this from data, not from medical textbooks — which is reassuring.

Lab slope features sometimes appear too, confirming that the trend features built in Cell 4.6 carry real signal.

---

## 📐 Section 6 — Evaluation (Choosing the Alarm Level)

### Cell 6.1 — The Official PhysioNet Utility Function

```python
def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3, ...):
```

**What it does:** This is the **official scoring formula** from the PhysioNet 2019 competition, copied verbatim from the competition's GitHub repository. It calculates how clinically valuable each alarm decision is, hour by hour.

The rules are:
- **Alarm fired up to 6 hours before sepsis onset → maximum reward (+1)**
- **Alarm fired 6–12 hours before onset → smaller reward (linearly declining)**
- **Alarm fired just after onset (up to 3 hours late) → small positive reward (declining)**
- **Alarm on a healthy patient → small penalty (−0.05 per hour)**
- **Missed sepsis patient (no alarm within the window) → large penalty (−2)**
- **No alarm, no sepsis → neutral (0)**

The final score is normalised so that:
- **1.0** = perfect (alarm at exactly the right time for every patient)
- **0.0** = doing nothing (never alarming)
- **< 0** = actively harmful (worse than silence)

A **self-test** at the end verifies the function is correctly copied: `compute_prediction_utility([0,0,0,0,1,1], [0,0,1,1,1,1])` should equal exactly 3.388889. If this assertion fails, something is wrong.

---

### Cell 6.2 — Load Saved Predictions and Ensemble Weights

```python
val_preds_df  = pd.read_csv(f'{ARTIFACT_DIR}/val_predictions.csv')
test_preds_df = pd.read_csv(f'{ARTIFACT_DIR}/test_predictions.csv')
```

**What it does:** Loads the previously saved probability scores for each patient-hour, and recomputes the ensemble probability to make sure it matches the stored weights exactly.

---

### Cell 6.3 — 📊 Figure: The Official Utility Reward Curve

```python
ax.plot(t_range, tp_utility, color='#01696f', lw=2.5, label='TP utility')
```

**What it does:** Draws a visual explanation of how the utility scoring system works — purely for the audience's understanding.

**🖼️ What the figure shows:**

The x-axis is **hours relative to sepsis onset** (0 = onset, negative numbers = before onset, positive = after onset). The y-axis is the score contribution.

- The green/teal shaded area is the **reward zone**: alarms here earn positive points.
- The pink/purple shaded area is the **penalty zone**: alarms here lose points.
- The green curve rises from 0 at the "earliest benefit" point (−12 hours from onset) to its peak of +1 at the optimal alarm time (−6 hours from onset), then falls back to 0 at the "latest benefit" point (+3 hours after onset).
- The horizontal dotted line shows the false alarm penalty: −0.05 per unnecessary alarm.

Annotation arrows label each key landmark. A clinician can look at this and immediately understand what the scoring system values: **catch it early, don't be too late, don't cry wolf.**

---

### Cell 6.4 — 📊 Figure: Threshold Sweep

```python
for thr in np.linspace(0.01, 0.99, 100):
    u = compute_physionet_utility_cohort(val_preds_df, PRIMARY_COL, thr)
```

**What it does:** The Ensemble model outputs a number between 0 and 1 for each patient-hour. We still need to decide at what point we shout "ALARM!". This cell tries 100 different cutoffs (from 0.01 to 0.99) and measures the official utility score for each.

**🖼️ What the figure shows:**

The x-axis is the threshold value (0.01 → 0.99). The y-axis is the resulting utility score.

The curve forms a **hump shape**:
- Very low threshold (e.g., 0.01): the model alarms almost constantly → loads of false alarms → negative utility (worse than doing nothing).
- Very high threshold (e.g., 0.95): the model almost never alarms → misses almost all real cases → near-zero utility (same as doing nothing).
- The **sweet spot** in the middle is where the alarm is selective enough to avoid fatigue but sensitive enough to catch real cases.

A pink dashed vertical line marks the **chosen threshold** — the one that achieved the highest utility on the validation set. A dot and annotation show the exact score at that peak. The blue shaded area highlights all thresholds where the model beats "doing nothing" (utility > 0).

This chosen threshold is saved and used in all subsequent test set evaluations.

---

### Cell 6.5 — 📊 Figure: Model Utility Comparison (Validation)

```python
ax.bar(results_df['Model'], results_df['Utility'], ...)
```

**What it does:** Uses the official scorer to evaluate all five models on the validation set at the chosen threshold, then draws a bar chart.

**🖼️ What the figure shows:**

Five bars, one per model. The height of each bar is the official utility score on the validation set.

- The **Ensemble bar** (purple) should be the tallest — it's the primary model.
- The **qSOFA bar** (clinical baseline) is often negative — meaning the simple rule is clinically *harmful* at scale due to its high false alarm rate.
- Every ML model bar above 0 means that model adds clinical value over doing nothing.

Numbers are written above each bar so the exact values are readable. This is the clearest single picture showing "ML beats the current clinical standard."

---

### Cell 6.6 — 📊 Figure: All-Model Threshold Sweep Overlay

```python
for model_name, col in sweep_cols:
    curve = [compute_physionet_utility_cohort(val_preds_df, col, t) ...]
    ax.plot(thresholds_sweep, curve, ...)
```

**What it does:** Repeats the threshold sweep from Cell 6.4, but overlays all four ML models on the same chart, plus a horizontal line for qSOFA (which doesn't have a continuous threshold).

**🖼️ What the figure shows:**

Four coloured curves (LR, XGBoost, LightGBM, Ensemble) plus a dotted horizontal line for qSOFA. You can see:
- The Ensemble (solid purple, thicker) generally has the highest or widest hump.
- The qSOFA line is often below 0 (harmful territory).
- The chosen threshold (vertical dashed line) is drawn based on where the Ensemble peaks.

This answers the question: "Is the chosen threshold a good choice for the other models too?" If all curves peak near the same point, the threshold is robust.

### Cell 6.7 — Summary

```python
print(f"  Chosen threshold : {BEST_THRESHOLD:.4f}")
print(f"  Validation score : {BEST_VAL_UTIL:.4f}")
```

**What it does:** Prints a clean summary of Section 6 and lists all the files saved to `artifacts/`. It also reminds you that the test set hasn't been touched yet — that happens in Section 7.

---

## 📊 Section 7 — Results (The Real Test)

### Loading Test Predictions

```python
test_preds = pd.read_csv(f'{ARTIFACT_DIR}/test_predictions.csv')
BEST_THRESHOLD = joblib.load(f'{ARTIFACT_DIR}/best_threshold.joblib')['threshold']
```

**What it does:** Loads the test set predictions and the threshold chosen in Section 6. This is the first time the test set is used for any actual evaluation.

---

### The Main Results Table

```python
models_eval = {
    'qSOFA Rule':          ('qsofa_score',    1.0),
    'Logistic Regression': ('lr_prob',        BEST_THRESHOLD),
    'XGBoost':             ('xgb_prob',       BEST_THRESHOLD),
    'LightGBM':            ('lgb_prob',       BEST_THRESHOLD),
    'Ensemble (primary)':  ('ensemble_prob',  BEST_THRESHOLD),
}
```

**What it does:** Evaluates all five models on the held-out test set using five metrics: AUPRC, AUROC, official utility score, sensitivity, and false alarm rate.

**How to read the output table:**
- **AUPRC** — above 0.018 (the random baseline) = model is learning something real
- **AUROC** — above 0.75 matches the published literature for this dataset
- **Utility (Official)** — above 0 = clinically better than doing nothing
- **Sensitivity** — what fraction of septic patients did we catch?
- **False Alarm Rate** — what fraction of healthy hours did we unnecessarily alarm?

---

### 📊 Figure: Precision-Recall Curves (Test Set)

```python
for name, (col, _) in models_eval.items():
    prec, rec, _ = precision_recall_curve(test_preds['label'], test_preds[col])
```

**What it does:** Same as the validation PR curves from Section 5, but now on the held-out test set.

**🖼️ What the figure shows:**

The five coloured lines represent the five models. The Ensemble (solid purple, thickest line) should sit highest, confirming it performs best on data it has never seen. Each label includes the AUPRC score. The grey dashed baseline shows what random guessing achieves (~0.017).

This is the primary proof that the models generalise — not just memorised the training data.

---

### Stratified Results by Eligibility Group

```python
for grp in ['never_septic', 'early_warning_eligible', 'immediate_only']:
    sub = test_preds[test_preds['eligibility_group'] == grp]
```

**What it does:** Breaks down the results separately for each of the three patient groups. This is important for honest reporting.

**Why it matters:** The `immediate_only` group's utility score will always be near 0 or negative — not because the model is bad, but because it's **structurally impossible** to warn early. If we mixed this group into the overall score, the model would look worse than it really is for the patients it *can* help.

---

### Hospital Subgroup Analysis

```python
for hosp in ['A', 'B']:
    sub = test_preds[test_preds['hospital_source'] == hosp]
```

**What it does:** Checks whether the model performs similarly at Hospital A and Hospital B.

**Why this matters:** If AUPRC is 0.09 at Hospital A but 0.04 at Hospital B, that's a red flag — the model learned something specific to Hospital A that doesn't generalise. Similar performance across hospitals is evidence the model learned general sepsis patterns, not hospital-specific quirks.

---

### Lead Time Analysis

```python
def compute_lead_times(df, prob_col, threshold):
    ...
    leads.append(onset - first_alarm)
```

**What it does:** For every `early_warning_eligible` patient the model correctly caught before onset, measures how many hours before onset the first alarm fired. Then reports summary statistics.

**Output:**
```
Median lead time : 7 hours
Mean lead time   : 9.1 hours
Catch rate       : 72%
```

A median lead time of 7 hours means: of the septic patients the model correctly caught early, the typical warning came 7 hours before sepsis was officially diagnosed. That's an actionable window — enough time for antibiotics and fluids to make a difference.

---

### 📊 Figure: Lead Time Histogram

```python
ax.hist(lt['leads'], bins=30, color='#a12c7b', ...)
ax.axvline(lt['median_lead_time'], ...)
ax.axvline(6, color='#964219', label='6h clinical threshold')
```

**What it does:** Draws a bar chart showing the distribution of lead times.

**🖼️ What the figure shows:**

The x-axis is "how many hours before sepsis onset did the alarm fire?" The y-axis is "how many patients had that lead time?"

- The tallest bars show the most common lead times.
- A green dashed vertical line marks the **median** lead time.
- An orange dotted vertical line marks the **6-hour clinical threshold** — the minimum advance warning considered clinically actionable.

You want most patients to be to the **right** of the orange line (warned more than 6 hours early). The percentage label in the title tells you exactly how many patients achieved this.

---

## 🔍 Section 8 — Error Analysis

### Cell 8.1 — Load Test Predictions

```python
test_preds = pd.read_csv(f'{ARTIFACT_DIR}/test_predictions.csv')
test_preds['pred_label'] = (test_preds['xgb_prob'] >= BEST_THRESHOLD).astype(int)
```

**What it does:** Loads the predictions and adds a binary alarm column — 1 if the model alarmed, 0 if it didn't.

---

### Cell 8.2 — Finding the Four Case Study Patients

```python
def find_success(df): ...    # caught before onset
def find_miss(df): ...       # had history, model never alarmed
def find_fa(df): ...         # healthy patient, model alarmed anyway
def find_struct(df): ...     # onset within first 6 hours
```

**What it does:** Searches the test set for four specific types of patient to illustrate the model's behaviour in different scenarios.

---

### Cell 8.3 — 📊 Figure: Utility Diagram

```python
ax.bar(hours, rewards, color=colors, ...)
```

**What it does:** A simpler, more visual version of the utility scoring explanation from Section 6.

**🖼️ What the figure shows:**

A bar chart where each bar represents one hour relative to sepsis onset (x-axis). The bar's height and colour show the score contribution:
- **Blue bars (positive):** Hours where an early alarm earned points.
- **Purple bar (−2.0):** The massive penalty for missing sepsis at the onset hour.
- **Orange bars (−0.05):** Small penalty for alarms fired too early.

This is a "reward chart" you can show to a clinical audience: "This is how the computer grades itself for each alarm decision."

---

### Cell 8.4 — 📊 Figure: Four Clinical Case Studies

```python
cases = [
    (success, '✅ SUCCESS\n(Caught before onset)', '#01696f'),
    (miss,    '❌ MISS\n(Had history, not flagged)', '#964219'),
    (false_alm,'🚨 FALSE ALARM\n(Never septic)', '#a12c7b'),
    (struct_lm,'🧱 STRUCTURAL LIMIT\n(Onset too early)', '#7a39bb'),
]
```

**What it does:** Plots four pairs of graphs (probability over time + respiratory rate over time) for four different real patients from the test set.

**🖼️ What the figure shows:**

Eight small plots arranged in 4 columns, 2 rows. Each column is one patient case:

**Column 1 — ✅ SUCCESS (green):**
The top graph shows the model's probability rising steadily over several hours, crossing the threshold (grey dashed line) well before the purple onset marker. The bottom graph shows respiratory rate climbing toward the 22 bpm mark. This is what a good early warning looks like.

**Column 2 — ❌ MISS (orange):**
The patient had a long ICU stay but the probability never crossed the threshold before onset. The respiratory rate may not have shown the classic pattern, or other vitals were confusing the model. This illustrates how atypical presentations escape detection.

**Column 3 — 🚨 FALSE ALARM (purple/red):**
A healthy patient — no sepsis ever — but the model's probability crossed the threshold repeatedly. The respiratory rate was chronically elevated (perhaps from COPD or anxiety), and the model couldn't distinguish septic from non-septic causes without richer clinical context.

**Column 4 — 🧱 STRUCTURAL LIMIT (violet):**
Sepsis onset happened within the first 1–2 hours of the ICU stay. There was no history to look at, no trend to detect. No model in the world could have warned here — this is not a failure, it's a fundamental constraint.

---

## ✅ Section 9 — Conclusion

### Cell 9.1 — Final Results Table

```python
results_df = pd.read_csv(f'{ARTIFACT_DIR}/model_comparison.csv')
```

**What it does:** Loads the final test-set results (already computed in Section 7) and prints them cleanly without recomputing anything.

**What the output means:**

The table shows all five models ranked by utility score. The key finding: every ML model has a higher utility score than qSOFA (the current clinical standard). The Ensemble achieves the highest AUPRC, meaning it's the best at ranking patients by sepsis risk. The model provides actionable advance warning for most of the patients it *can* warn (the `early_warning_eligible` group).

---

## 🎯 Section 10 — Live Demo

### Cell 10.1 — Load Saved Models

```python
xgb_loaded = joblib.load(f'{ARTIFACT_DIR}/xgb_model.joblib')
lgb_loaded = joblib.load(f'{ARTIFACT_DIR}/lgb_model.joblib')
lr_loaded  = joblib.load(f'{ARTIFACT_DIR}/lr_model.joblib')
```

**What it does:** Loads the three saved models from disk. This is very fast (a few seconds). No retraining happens.

---

### Cell 10.2 — Select Demo Patients

```python
for grp in ['early_warning_eligible', 'immediate_only', 'never_septic']:
    demo_pids.extend(pids[:2])
```

**What it does:** Picks 6 real test patients — 2 from each eligibility group — to demonstrate live inference.

---

### Cell 10.3 — LIVE Inference

```python
live_xgb_probs = xgb_loaded.predict_proba(X_demo)[:, 1]
live_lgb_probs = lgb_loaded.predict_proba(X_demo)[:, 1]
live_lr_probs  = lr_loaded.predict_proba(X_demo)[:, 1]
live_ens_probs = xgb_w * live_xgb_probs + lgb_w * live_lgb_probs + lr_w * live_lr_probs
live_alarms    = (live_ens_probs >= THRESHOLD_LOADED).astype(int)
```

**What it does:** This is the dramatic proof that the models work. The three saved models are applied to the 6 demo patients **right now**, in front of the audience. The probabilities are freshly calculated — not copied from a file.

The ensemble combines all three into a final probability, and the threshold converts it into a binary alarm decision.

**Output example:**
```
✅ LIVE inference on 342 patient-hours
   Primary model : Ensemble (XGB 45% + LGB 45% + LR 10%)
   Threshold     : 0.1200
   Alarm rate    : 18.42%
```

---

### Cell 10.4 — Show Predictions Table

```python
print(sub[['patient_id', 'ICULOS', 'label', 'live_ensemble_prob', 'live_alarm']].head(12))
```

**What it does:** Prints the first 12 rows of predictions for each of the 3 demo patients, showing the ensemble probability and whether the alarm fired at each hour.

---

### Cell 10.5 — 📊 Figure: Live Demo Trajectory

```python
ax1.plot(pdf['ICULOS'], pdf['live_ensemble_prob'], color='#a12c7b', ...)
ax1.plot(pdf['ICULOS'], pdf['live_xgb_prob'],      color='#4f98a3', ls='--', ...)
```

**What it does:** Draws a two-panel chart for one demo patient showing the live inference results.

**🖼️ What the figure shows:**

**Top panel (Probability):**
- The thick **purple solid line** is the Ensemble's probability over the patient's ICU stay.
- The thin **blue dashed line** is XGBoost's probability (shown for reference).
- The **grey dashed horizontal line** is the alarm threshold.
- The **pink shaded region** shows where the Ensemble is above the threshold — i.e., the alarm is ON.
- If the patient is septic, an **orange vertical dashed line** marks the onset hour.

Ideally, the pink region starts several hours to the left of the orange line — proof of early warning.

**Bottom panel (Respiratory Rate):**
- The orange line tracks breathing rate over time.
- A grey dotted line marks the clinical qSOFA threshold of 22 breaths per minute.
- The orange vertical onset marker (same as above) helps correlate clinical signs with model output.

This is the most visual, intuitive demonstration of what the whole notebook achieves: a computer watching a patient's vital signs and raising an alarm before the doctors even know something is wrong.

---

## 📁 Artifacts Produced

After a full training run, the `artifacts/` folder contains:

| File | What It Is |
|---|---|
| `xgb_model.joblib` | Trained XGBoost model |
| `lgb_model.joblib` | Trained LightGBM model |
| `lr_model.joblib` | Trained Logistic Regression pipeline |
| `ensemble_weights.joblib` | Saved XGB/LGB/LR mix proportions |
| `best_threshold.joblib` | The alarm threshold chosen in Section 6 |
| `feature_cols.json` | The exact list of 150+ features used |
| `val_predictions.csv` | Validation set probabilities (all models) |
| `test_predictions.csv` | Test set probabilities (all models) |
| `eligibility.csv` | Patient eligibility group labels |
| `model_comparison.csv` | Final test-set results table |
| `stratified_results.csv` | Results by eligibility group |
| `hospital_subgroup.csv` | Results by hospital |
| `lead_time_summary.json` | Lead time statistics for the Ensemble |
| `label_distribution.png` | Class imbalance bar charts |
| `missingness.png` | Lab missingness bar chart |
| `sample_trajectories.png` | Sample patient vital sign plots |
| `pr_curve_comparison.png` | Validation PR curves (all models) |
| `feature_importance.png` | XGBoost top-15 feature importance |
| `utility_curve_explanation.png` | Official utility reward diagram |
| `threshold_sweep.png` | Utility vs threshold (Ensemble) |
| `threshold_sweep_all_models.png` | Utility vs threshold (all models) |
| `val_utility_comparison.png` | Bar chart: model utilities (validation) |
| `pr_curves.png` | Test set PR curves (all models) |
| `lead_time_histogram.png` | Lead time distribution histogram |
| `utility_diagram.png` | Simplified utility reward bar chart |
| `error_analysis.png` | Four clinical case study plots |
| `demo_trajectory.png` | Live demo trajectory chart |

---

## 🧠 Key Concepts Glossary

| Term | Simple Explanation |
|---|---|
| **Sepsis** | A life-threatening condition where the body's response to infection harms itself |
| **ICU** | Intensive Care Unit — the most serious ward in a hospital |
| **Onset** | The hour when sepsis is officially diagnosed (`SepsisLabel` flips from 0 to 1) |
| **AUPRC** | A measure of model quality under imbalanced data — higher is better; random baseline ≈ 0.018 |
| **AUROC** | A threshold-free measure of how well the model ranks patients — 0.5 = random, 1.0 = perfect |
| **Utility Score** | The official clinical score: above 0 = better than silence, 1.0 = perfect |
| **False Alarm Rate** | How often the alarm fires when nothing is wrong |
| **Lead Time** | How many hours before sepsis onset the alarm fired |
| **Threshold** | The minimum probability the model must output before it shouts "ALARM!" |
| **Ensemble** | Combining multiple models — here, weighted average of XGBoost + LightGBM + LR |
| **Data Leakage** | When the model secretly sees test data during training, making it look falsely good |
| **Forward Fill** | Carrying the last known value forward in time to fill gaps |
| **Slope Feature** | A measure of whether a vital sign is going up or down over the last 6 hours |
| **Optuna** | A tool that automatically searches for the best model settings |
| **FORCE_RETRAIN** | A switch: `True` = train fresh (~10 min), `False` = load from disk (~60 sec) |
| **qSOFA** | A simple 3-criteria clinical rule used at the bedside today — the baseline to beat |

---

*README generated for Group ML_SE_ALL_02 — Machine Learning Summer 2026*