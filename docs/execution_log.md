# Project Execution Log & Validation Tracking

This log tracks every step executed, along with the validation commands and criteria used to verify correctness.

---

## Log Entries

### Milestone 0: Documentation Setup
- **Timestamp**: 2026-10-11
- **Action**: Created initial documentation structure inside [`docs/`](file:///c:/Users/anush/OneDrive/Desktop/IIT/AI/AI-Powered%20Cybersecurity%20Threat%20Detection/docs).
- **Status**: Completed.

### Milestone 1: Repository Foundation & Structural Validation
- **Timestamp**: 2026-10-11
- **Action**: Established project tree, starter modules, documentation index, and initial tests.
- **Status**: Completed.

### Milestone 2: Dataset Ingestion, Validation & Preprocessing
- **Timestamp**: 2026-10-11
- **Action**: Implemented CSV loading, column normalization, non-finite handling, target label validation, and initial feature/target separation.
- **Status**: Completed.

### Milestone 3: Feature Engineering, Model Training & Evaluation Pipeline
- **Timestamp**: 2026-10-11
- **Actions**:
  1. Verified environment: Python 3.10.8, Pandas 2.1.4, NumPy 1.26.4, Scikit-learn 1.3.2, Pytest 9.1.1, Joblib 1.5.3.
  2. Confirmed dataset status: `data/raw/` contains only `.gitkeep`. No actual CIC-IDS2017 files present.
  3. Implemented `src/feature_engineering.py`:
     - `FeatureSchema` dataclass validating feature presence, ordering, and numeric types at inference time.
     - `prepare_feature_matrix()` excluding target and metadata columns (Flow ID, IPs, Timestamps) to prevent leakage.
     - Strict rejection of duplicate column names and arbitrary non-numeric coercion.
  4. Implemented `src/preprocessing.py` enhancements:
     - `encode_binary_target()` mapping Benign -> 0 and recognized attack classes -> 1, rejecting missing and unknown labels.
     - `train_test_split_network_flows()` with stratified partitioning prior to transformer fitting.
     - `split_data_temporal()` implementing chronological splitting when timestamps are present.
  5. Implemented `src/train.py`:
     - `build_training_pipeline()` combining `SimpleImputer(strategy='median')` and `RandomForestClassifier(class_weight='balanced')`.
     - `train_model()` fitting imputer strictly on `X_train` and recording comprehensive training metadata.
     - `save_model_artifact()` with explicit overwrite protection (`overwrite=False` by default).
     - `load_model_artifact()` for safe deserialization and structure verification.
  6. Implemented `src/evaluate.py`:
     - `compute_binary_metrics()` computing Accuracy, Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, and Confusion Matrix.
     - Strict positive class semantics (`pos_label=1`) and dynamic probability index resolution.
     - `save_evaluation_report()` exporting formatted JSON reports.
     - Optional confusion matrix visualization plot generation via matplotlib.
  7. Implemented `src/predict.py`:
     - `load_model_for_inference()` and `predict_network_flows()` with `FeatureSchema` alignment and probability extraction.
  8. Expanded test suite to 43 unit and regression tests across 6 test modules:
     - All 43 tests passed in 4.02s without requiring CIC-IDS2017.
  9. Executed end-to-end synthetic smoke test validating full ingestion -> preparation -> split -> train -> evaluate -> save/reload -> predict pipeline.
  10. Synchronized documentation: `README.md`, `docs/methodology.md`, `docs/data_pipeline.md`, `docs/model_design.md`.
- **Validation**: 43 pytest tests passed; syntax compileall passed with 0 errors; synthetic smoke test passed.
- **Status**: Completed.

### Milestone 4: CLI Orchestration, Batch Prediction, Threat Alerts & Edge-Case Validation
- **Timestamp**: 2026-10-11
- **Actions**:
  1. Inspected all modules, function signatures, dependencies, and verified `data/raw/` contains only `.gitkeep`.
  2. Fixed and tested critical ML edge cases:
     - Strict label whitelisting in `src/preprocessing.py` with case/whitespace trimming, rejecting empty/unknown/unexpected labels.
     - Dynamic positive class probability resolution from `model.classes_` in `src/predict.py` and `src/evaluate.py`, preventing reverse-class index bugs and checking finite bounds `[0, 1]`.
     - Explicit rejection of 100% missing numeric feature columns in `src/feature_engineering.py`.
     - Robust temporal splitting in `src/preprocessing.py` with datetime parsing, missing/invalid timestamp rejection, and minimum sample checks.
  3. Implemented full command-line orchestrator in `main.py` using `argparse`:
     - `train`: Train pipeline on labeled CSV, compute held-out metrics, save pipeline artifact and evaluation report JSON with overwrite protection.
     - `evaluate`: Evaluate saved model on labeled CSV, report confusion matrix, accuracy, precision, recall, F1, FPR, ROC-AUC without retraining.
     - `predict`: Predict network flows from unlabeled/feature CSV, produce output CSV with predicted classes and attack scores, extract structured alert events for attacks.
     - Operational logging to `outputs/logs/cli.log`.
  4. Connected and extended alert engine in `src/alert_engine.py`:
     - Added `process_batch_alerts()` generating structured alert objects only for attack predictions without fabricating network artifacts.
     - Added `save_alerts_json()` with overwrite protection.
  5. Implemented comprehensive test suite in `tests/test_cli.py` and expanded unit tests across all modules to 63 total automated tests.
  6. Executed Phase 7 validation:
     - `python -m compileall src tests app.py main.py`: 0 errors.
     - `python -m pytest tests/ -v`: 63 passed in 4.78s.
     - Full synthetic end-to-end test workflow (train -> evaluate -> predict -> alerts -> negative missing model -> negative invalid labels) executed in isolated temporary directory.
     - Production directory verification: `models/` and `outputs/reports/` kept clean of synthetic artifacts.
- **Validation**: 63 passed pytest tests; compileall passed; synthetic CLI workflows verified.
- **Status**: Completed.

### Milestone 5A: Integration Audit, Alert Severity Verification & Streamlit Dashboard
- **Timestamp**: 2026-10-11
- **Actions**:
  1. Audited evaluation integrity across `main.py`, `src/train.py`, `src/evaluate.py`, and `src/predict.py`:
     - Confirmed training metrics are strictly evaluated on held-out test splits.
     - Confirmed evaluation mode never retrains.
     - Added metadata tracking (`evaluation_dataset`, `evaluation_type`, and cross-dataset overlap disclaimer) to evaluation reports.
     - Enforced binary label validation in `compute_binary_metrics()` and explicit handling of undefined ROC-AUC when ground truth has a single class.
  2. Audited alert severity engine (`src/alert_engine.py`):
     - Documented probability triage severity tiers: BENIGN (<0.50), LOW ([0.50, 0.70)), MEDIUM ([0.70, 0.85)), HIGH ([0.85, 0.95)), CRITICAL ([0.95, 1.00]).
     - Handled boundary values, missing scores (`None`, `NaN`, `Inf` -> "UNKNOWN"), out-of-range inputs (raising `ValueError`), and added prototype triage disclaimer.
  3. Implemented full interactive Streamlit dashboard in `app.py`:
     - **Overview Tab**: Session triage metrics, model readiness check, operational disclaimers, and empty state handling.
     - **Batch Flow Prediction Tab**: Drag-and-drop CSV upload, preview, `FeatureSchema` validation with plain-language errors, decision threshold slider, interactive filtering, and direct CSV/JSON downloads.
     - **Threat Alerts Feed Tab**: Tabular display of alerts with Flow IDs, severity tiers, timestamps, and genuine telemetry metadata.
     - **Model & Evaluation Audit Tab**: Model hyperparameter inspection, expected feature schema list, and verified evaluation report benchmarks.
  4. Created test suite in `tests/test_dashboard.py` and expanded evaluation/alert engine tests:
     - 77 total automated tests passing.
  5. Verified headless Streamlit startup via `streamlit run app.py --server.headless true --server.port 8501`.
- **Validation**: 77 passed pytest tests; compileall passed; Streamlit server launch verified.
- **Status**: Completed.

### Milestone 5B: Real Dataset Acquisition Documentation & Memory-Aware Validation
- **Timestamp**: 2026-10-11
- **Actions**:
  1. Documented official CIC-IDS2017 dataset acquisition from the University of New Brunswick (UNB) portal (https://www.unb.ca/cic/datasets/ids-2017.html) in `docs/dataset.md` and `README.md`.
  2. Documented placement of the 8 extracted network flow CSV files in `data/raw/` with disk (~2.8–3.1 GB) and RAM (~6–12 GB) projections explicitly labeled as estimates.
  3. Implemented `src/dataset_validator.py` providing memory-aware, chunked streaming inspection (`validate_single_csv` and `validate_dataset_directory`):
     - Profiles file sizes, row counts, dtypes, missing cells, positive and negative infinities.
     - Profiles label distributions, frequencies, percentages, and flags unknown labels against canonical whitelists.
     - Identifies timestamp formatting failures (NaT count).
     - Tracks duplicate flow rows using 64-bit integer row hash sets.
     - Evaluates schema compatibility against model `FeatureSchema`.
     - Analyzes target and session leakage risks (identifiers, timestamps, duplicate flows).
     - Explicitly guarantees that no model training or evaluation occurs (`model_training_performed=False`).
  4. Integrated `validate-data` command into `main.py` CLI supporting both positional (`python main.py validate-data ...`) and flag (`python main.py --mode validate-data ...`) invocations.
  5. Created comprehensive test suite in `tests/test_dataset_validator.py` covering 17 distinct scenarios (94 total tests passing across test suite).
  6. Executed validation against actual `data/raw/` directory: correctly detected absence of CSV files (`status: NO_FILES_FOUND`), produced report at `outputs/reports/dataset_validation.json`, and exited with code 1.
- **Validation**: 94 passed pytest tests in 6.12s; compileall passed with 0 errors; real data/raw validation executed.
- **Status**: Completed.



