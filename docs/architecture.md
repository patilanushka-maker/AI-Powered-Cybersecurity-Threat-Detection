# System Architecture & Technical Specifications

## 1. High-Level System Architecture

The **AI-Powered Cybersecurity Threat Detection System** processes network flow captures, engineers behavioral attributes, evaluates risk via machine learning classifiers, and triggers actionable security alerts.

```
+-----------------------------------------------------------------------------------+
|                              1. DATA SOURCES & INGESTION                          |
|  Raw Network Flow CSVs (CIC-IDS2017)  --->  Ingestion & Schema Validation Module  |
+---------------------------------------------------------+-------------------------+
                                                          |
                                                          v
+-----------------------------------------------------------------------------------+
|                        2. PREPROCESSING & FEATURE ENGINEERING                     |
|  Header Cleaning & Inf/NaN Handling  --->  Encoding & Scaling  --->  Feature Selection |
+---------------------------------------------------------+-------------------------+
                                                          |
                       +----------------------------------+----------------------------------+
                       |                                                                     |
                       v                                                                     v
+----------------------------------------------------+  +------------------------------------+
|               TRAINING WORKFLOW (OFFLINE)          |  |           INFERENCE WORKFLOW       |
+----------------------------------------------------+  +------------------------------------+
|  * Stratified Train / Val / Test Partitioning      |  |  * Streaming / Batch Flow CSVs     |
|  * Fit Preprocessor on Train Split only            |  |  * Transform via Saved Pipeline    |
|  * Train Baseline (Logistic Regression)            |  |  * Generate Class & Probabilities  |
|  * Train Primary Model (Random Forest)             |  |  * Alert Engine (Rule/Thresholds)  |
|  * Offline Evaluation (F1, Recall, FPR, ROC)       |  |  * Streamlit Dashboard / Reports   |
|  * Persist Serialized Artifacts (.joblib)          |  |  * Structured Output Logging       |
+----------------------------------------------------+  +------------------------------------+
```

---

## 2. End-to-End Data Flow

The architecture strictly follows this sequential pipeline:
1. **CSV Input**: Labeled network flow captures in CSV format.
2. **Ingestion & Validation**: Schema validation, header stripping, empty file detection, format type assertions.
3. **Preprocessing**: Handling `NaN` and `Infinity` values, removing duplicate records, timestamp/flow alignment.
4. **Feature Selection**: Filtering zero-variance columns, correlation pruning, domain-specific network traffic indicators.
5. **Model Training / Evaluation**: Offline training using stratified cross-validation and held-out test splits.
6. **Saved Model**: Serialized models and fitted preprocessors saved with integrity metadata to `models/`.
7. **Prediction**: Inference module applying preprocessor transformations to new flows to generate intrusion probability scores.
8. **Alert Generation**: Security alert engine categorizing severity (Low, Medium, High, Critical) based on prediction thresholds and attack classes.
9. **Dashboard / Output Export**: Interactive Streamlit interface and structured exports to `outputs/reports/`, `outputs/predictions/`, and `outputs/logs/`.

---

## 3. Separation of Workflows: Training vs. Inference

A core architectural principle is the rigorous separation between the **Training Workflow** and the **Inference Workflow**:

### 3.1 Training Workflow (Offline Batch)
- Operates on historic labeled CSVs.
- Fits transformers (imputers, scalers, encoders) **exclusively on the training split** to prevent data leakage.
- Performs model selection, cross-validation, and hyperparameter tuning.
- Saves fitted artifacts (`models/preprocessor_pipeline.joblib`, `models/random_forest_baseline.joblib`).
- Generates validation reports in `outputs/reports/`.

### 3.2 Inference Workflow (Online / Batch Scoring)
- Operates on unlabeled or newly observed network flows.
- Loads existing frozen preprocessor and model artifacts in read-only mode.
- Performs feature transformations using `.transform()` (never `.fit()` or `.fit_transform()`).
- Yields predictions, confidence percentiles, and alerts.
- Emits real-time logs to `outputs/logs/` and visualizations to the Streamlit UI.
