# 🎙️ Master Team Presentation Talk & Speaker Coaching Guide

This document is written as a **complete, word-for-word spoken script and explanation guide** designed to help every team member explain their section with absolute mastery and confidence. 

Instead of memorizing dry bullet points, read through your conversational script below. It explains **why** we made each decision in plain, natural language so you sound like an experienced Machine Learning Engineer during the talk and Q&A.

---

## ⏱️ Presentation Schedule & Roles

| Team Member | Allocated Time | Core Topics Owned | Notebook Section to Display |
|---|---|---|---|
| **Speaker 1** | 0:00 — 1:15 | Introduction, Sepsis Problem & Dataset Bottlenecks | Cell 0 (Title) → Section 1 → Section 2 (EDA Charts) |
| **Speaker 2** | 1:15 — 2:30 | Clinical Preprocessing, 6h Windows & Patient Split | Section 3 (Pipeline) → Section 4 (Cell 42 Split Check) |
| **Speaker 3** | 2:30 — 3:45 | Model Selection, Utility Reward & Threshold Sweep | Section 5 (Models) → Section 6 (Utility Curve) |
| **Speaker 4** | 3:45 — 5:00 | Test Results, 4 Error Cases & Live Demonstration | Section 7 (Results) → Section 8 → Section 10 (Demo) |

---

## 📢 Detailed Spoken Script & Concept Explanations

### 👤 Speaker 1: The Medical Problem & Data Landscape (0:00 — 1:15)

#### 🎬 Stage Cue:
Open the notebook at **Cell 0 (Title Slide)**. As you start speaking, smoothly scroll down to **Section 1 (Problem Statement)** and then **Section 2 (Exploratory Data Analysis)** so the judges see the label distribution and missingness charts.

#### 🗣️ Word-for-Word Spoken Script:
> "Good morning respected judges, professors, and guests. We are Team ML_SE_ALL_02, and we are proud to present our applied machine learning project: **Early Sepsis Warning with Utility-Based Clinical Thresholds**.
> 
> To understand our project, we first have to understand what happens in an Intensive Care Unit. Sepsis is a catastrophic organ failure caused by the body's extreme response to an infection. In the ICU, sepsis is a race against the clock — every hour that antibiotic treatment is delayed increases patient mortality by 7%. 
> 
> So why don't hospitals just set alarms to go off at the slightest sign of fever? Because of a dangerous medical phenomenon called **'Alarm Fatigue.'** Traditional hospital bedside monitors beep constantly with false alarms — up to 90% of them are false alerts. When monitors beep every two minutes, doctors and nurses get exhausted and subconsciously start ignoring them. Therefore, building a standard machine learning model that just optimizes for raw accuracy is completely useless here. We need a system that detects sepsis **hours before clinical onset**, while strictly keeping false alarms to a minimum.
> 
> For our analysis, we used the official **PhysioNet 2019 Challenge dataset**, which tracks over 40,000 ICU patients across two separate hospital systems, recording 40 physiological variables every hour. When we explored the data, we found two massive bottlenecks: First, extreme class imbalance — only **1.8%** of the hourly observations are septic. Second, severe data missingness — lab tests like Lactate and Blood Cultures are missing over 90% of the time. But as we will show you, our methodology turns these bottlenecks into diagnostic strengths.
> 
> I will now hand over to my colleague, Member 2, to explain how we preprocessed this data safely."

#### 💡 If the Professor Interrupts You: How to Explain Key Concepts
* **What is informative missingness?** *"Professor, in an ICU, vital signs like heart rate are measured automatically by machines every hour. But lab tests require a doctor to physically order a blood draw. If Lactate is missing 90% of the time, it means doctors didn't order it. When doctors suddenly order 3 blood tests in 6 hours, that cluster of orders is an active clinical signal that the patient is deteriorating!"*

---

### 👤 Speaker 2: Preprocessing, 6h Sliding Windows & Leakage Prevention (1:15 — 2:30)

#### 🎬 Stage Cue:
Scroll to **Section 3 (Methodology Pipeline Flowchart)** and **Section 4 (Preprocessing)**. When you mention the patient split, explicitly point your mouse cursor to **Cell 42**, highlighting the print statement that confirms `0 overlapping patients`.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 1. To process raw hospital records without corrupting our models, we designed a customized clinical preprocessing architecture. 
> 
> First, how do we handle missing vital signs? Standard data science textbooks tell you to use mean imputation — replacing missing values with the overall hospital average. In an ICU, that is fatal. If a patient's heart rate drops to 40, replacing a missing reading an hour later with the hospital average of 80 masks their collapse! Instead, we used **Clinical Forward-Filling**. We carry the patient's last measured value forward, reflecting the true clinical assumption that a patient's physiological state persists until a new measurement is taken.
> 
> Second, doctors don't diagnose patients by looking at an isolated single hour; they look at **trends**. Therefore, we engineered **6-hour sliding window features**. For every vital sign, we calculate the rolling 6-hour mean and fit a linear regression line to compute the **slope**. A patient with a static heart rate of 95 is stable. But a patient whose heart rate climbed from 70 to 95 over the past 6 hours has a steep positive slope — capturing acute deterioration long before static thresholds are breached. We also calculate lab missingness rates over the window, allowing the AI to learn test-ordering frequency as a feature.
> 
> *(Point to Cell 42)*
> Finally, we addressed the most critical trap in healthcare AI: **Time-Series Data Leakage**. Because hourly measurements from the same patient are heavily correlated, doing a standard random train/test split scatters hours from the same patient across both sets. The model would memorize the patient's identity rather than learning sepsis! To prevent this, we performed a strict **Patient-Level Split** using `GroupShuffleSplit`: 70% Train, 10% Validation, and 20% Test. As you can see right here in Cell 42, our automated code verification confirms exactly zero overlapping patients across our splits.
> 
> I will now turn it over to Member 3 to discuss our models and threshold calibration."

#### 💡 If the Professor Interrupts You: How to Explain Key Concepts
* **Why 6 hours for the window?** *"We tested different windows. A 2-hour window is too noisy due to temporary medication spikes or patient movement. A 12-hour window washes out acute changes. Six hours perfectly reflects a standard ICU nursing shift trend."*

---

### 👤 Speaker 3: Model Architecture & Threshold Calibration (2:30 — 3:45)

#### 🎬 Stage Cue:
Scroll to **Section 5 (Selected Models Table)** and then **Section 6 (Evaluation)**. Display the **Official PhysioNet Utility Matrix** and point to the **Threshold Sweep Graph (`threshold_sweep.png`)** showing the peak curve at 0.34.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 2. To evaluate our AI objectively, we first established a medical baseline using the hospital **qSOFA rule** — the standard manual checklist doctors use assessing respiration and blood pressure. We then trained three diverse algorithms: a linear ElasticNet Logistic Regression, gradient boosted XGBoost, and LightGBM tuned across 50 Optuna optimization trials. Finally, we blended them into a weighted **Soft-Vote Ensemble**, assigning 45% weight to XGBoost, 45% to LightGBM, and 10% to Logistic Regression for maximum stability.
> 
> But how do we score these models? If we used standard classification accuracy, a dumb model that predicts 'No Sepsis' for everyone would achieve 98.2% accuracy because only 1.8% of rows are septic! Instead, we embedded the official verbatim **PhysioNet Clinical Utility Scorer**. This custom reward function mirrors medical reality: it awards maximum reward (**+1.0 point**) if the model sounds an alarm **6 to 12 hours before** clinical onset, giving doctors enough lead time to act. If the warning sounds too late, the reward drops toward zero. And crucially, every false alarm receives an active penalty (**-0.05 points**) to punish alarm fatigue.
> 
> *(Point to Threshold Sweep Graph)*
> This leads to our most impactful technical contribution: **Threshold Calibration**. Standard machine learning models classify a positive result when probability crosses 0.50. Under extreme 1.8% class imbalance, a model rarely reaches 0.50 confidence, causing catastrophic miss rates. We ran a systematic threshold sweep strictly on our validation set and discovered that the optimal clinical decision cutoff is **0.34**. By lowering our alarm cutoff from 0.50 to 0.34, we capture deteriorating patients much earlier, boosting our clinical utility score by over **300%** without ever touching our test data.
> 
> Now, Member 4 will walk us through our final results and live code demonstration."

#### 💡 If the Professor Interrupts You: How to Explain Key Concepts
* **Why did you include Logistic Regression in an ensemble with tree models?** *"Tree models are incredible at non-linear vital interactions, but they can occasionally extrapolate poorly on extreme outliers. Logistic regression provides a smooth, regularized linear anchor that keeps the ensemble bounded and stable."*

---

### 👤 Speaker 4: Results, Error Analysis & Live Demo (3:45 — 5:00)

#### 🎬 Stage Cue:
Scroll to **Section 7 (Results Table)**, highlight **Section 8 (Error Analysis Case Studies)**, and then click into **Section 10**. Run the live demo cells to show the system predicting sepsis in real-time.

#### 🗣️ Word-for-Word Spoken Script:
> "Thank you, Member 3. When we evaluated our calibrated Ensemble on the completely unseen 20% Test set, the results were definitive. Our model achieved an official Clinical Utility Score of **0.327** and an AUPRC of **0.098**. To put that in perspective, our machine learning ensemble outperformed the hospital qSOFA baseline by **over 60-fold**, while giving doctors a median early warning lead time of **6.0 full hours** before clinical sepsis onset. Furthermore, our subgroup analysis confirmed robust generalizability across both Hospital A and Hospital B.
> 
> In Section 8, we went beyond raw numbers and performed clinical error analysis across 4 specific patient case studies: Success, Miss, False Alarm, and Structural Limitation. We made a profound discovery: roughly **25% of septic patients arrive at the ICU already septic on Hour 1**. It is a structural impossibility for any early warning monitor to give a 6-hour advance warning for a patient who enters the door already in septic shock. Being honest about this limitation explains why the maximum theoretical utility ceiling is around 0.65, not 1.0.
> 
> *(Execute Section 10 Cells)*
> Finally, we bring our code to life in Section 10. Notice our Run Control flag at the top is set to `FORCE_RETRAIN = False`, allowing our centralized loader to load our pre-trained artifacts instantly in fast-load mode. Watch as we stream an unseen test patient's trajectory hour-by-hour: everything is stable until ICU Hour 18, where rising heart rate and falling respiration slopes push our ensemble probability across the 0.34 threshold. The system triggers a high-priority sepsis alarm **exactly 6 hours before** clinical onset occurs at Hour 24. 
> 
> This concludes our presentation. Our code is clean, verifiable, and ready for deployment testing. Thank you, and we welcome your questions."

---

## 🛡️ Master Team Q&A Coaching Guide

When answering questions, **do not give one-word answers**. Use the proven explanations below:

| Professor's Question | Winning Team Explanation |
|---|---|
| **"Why is your AUPRC score around 0.10? Isn't 0.10 a terrible score?"** | "In standard balanced datasets, yes, but not under extreme 1.8% class imbalance! In AUPRC, a random random guess scores equal to the positive class prevalence — which is 0.018. Our score of 0.098 represents more than a **5-fold improvement over random baseline**, and is 5 times higher than the hospital qSOFA score (0.019)." |
| **"How do we know your Optuna tuning didn't just overfit to the validation data?"** | "We prevented overfitting through three layers: First, 5-fold cross-validation inside Optuna with early stopping. Second, strict separation of patients using GroupShuffleSplit. And third, empirical proof: our Utility score held completely stable when evaluated on the unseen 20% Test set (**0.327 Test vs 0.341 Validation**)." |
| **"What happens if I change `FORCE_RETRAIN = True` and execute Cell 10 right now?"** | "The centralized loader will bypass disk loading and trigger the full pipeline: it will load raw `.psv` files, compute forward fills and 6h rolling slopes, run 50 Optuna optimization trials per boosted tree model, search grid weights for the ensemble, re-calibrate the 0.34 threshold, and overwrite the saved artifacts. It completes smoothly in about 7 minutes." |
| **"Why did you include `immediate_only` patients if you know you can't predict them early?"** | "To preserve strict data integrity and clinical transparency. Filtering out patients who arrive already septic would artificially inflate our score to look better than reality. Instead, we report overall results honestly and explain the ~25% structural admission limitation in our error analysis section." |
