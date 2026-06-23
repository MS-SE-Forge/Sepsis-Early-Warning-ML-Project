# Sepsis Early Warning System - Architecture Diagram

The following diagram provides a detailed, modular overview of the data pipeline, feature engineering logic, machine learning lifecycle, and evaluation process.

```mermaid
flowchart TB
    %% Styling
    classDef rawData fill:#e2e8f0,stroke:#64748b,stroke-width:2px,color:#0f172a
    classDef etl fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a
    classDef model fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef eval fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#78350f
    classDef output fill:#f3e8ff,stroke:#a855f7,stroke-width:2px,color:#581c87

    %% Stage 1: Ingestion
    subgraph Stage1 [1. Data Ingestion & Setup]
        A1[(PhysioNet 2019 Dataset\ninput/)]:::rawData
        B1[01_eda.py\nRaw Data Loader]:::etl
        C1[08_eligibility_tagging.py\nCohort Stratification]:::etl
        
        A1 -->|Read .psv files| B1
        B1 -->|Assign Hospital Source| C1
    end

    %% Stage 2: Feature Engineering
    subgraph Stage2 [2. Feature Engineering & Split]
        A2[02_features_split.py\nData Transformations]:::etl
        B2(Missingness Indicators\nExplicit Flags):::etl
        C2(Forward Fill\nCarry-forward Vitals):::etl
        D2(6-Hour Rolling Windows\nMeans & Slopes):::etl
        E2[GroupShuffleSplit\nPatient-Level Isolation]:::etl
        
        C1 --> A2
        A2 --> B2
        B2 --> C2
        C2 --> D2
        D2 --> E2
    end

    %% Stage 3: Modeling
    subgraph Stage3 [3. Machine Learning Core]
        A3[03_baselines.py\nLogistic Reg & qSOFA]:::model
        B3[04_xgboost_main.py\nXGBoost Classifier]:::model
        C3(scale_pos_weight\nHandle Imbalance):::model
        D3(Optimize AUCPR\nPrecision-Recall Focus):::model
        
        E2 -->|Train/Val Data| A3
        E2 -->|Train/Val Data| B3
        B3 --> C3
        C3 --> D3
    end

    %% Stage 4: Evaluation
    subgraph Stage4 [4. Clinical Utility & Evaluation]
        A4[06_threshold_leadtime.py\nThreshold Tuning]:::eval
        B4(Max Approx Utility\nReward Lead Time):::eval
        C4[09_stratified_evaluation.py\nBy Eligibility Group]:::eval
        D4[10_hospital_subgroup.py\nHospital A vs B]:::eval
        E4[11_error_analysis.py\nQualitative Traces]:::eval
        F4[05_utility_eval_prep.py\nOfficial PhysioNet Scorer]:::eval
        
        D3 -->|Probabilities| A4
        A4 --> B4
        B4 --> C4
        B4 --> D4
        B4 --> E4
        B4 --> F4
    end

    %% Stage 5: Output & Presentation
    subgraph Stage5 [5. Artifact Orchestration]
        A5[12_build_artifacts.py\nMaster Pipeline]:::output
        B5[(artifacts/\nJoblib, JSON, CSV)]:::output
        C5[build_notebook.py\nNotebook Generator]:::output
        D5(sepsis_early_warning.ipynb\nFinal Analysis):::output
        
        Stage1 -.-> A5
        Stage2 -.-> A5
        Stage3 -.-> A5
        Stage4 -.-> A5
        A5 -->|Save State| B5
        B5 -->|Instant Load| C5
        C5 --> D5
    end
```
