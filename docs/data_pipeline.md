# Data Pipeline & Flow Specifications

## 1. End-to-End Pipeline Architecture (Milestones 2 & 3 Implemented)

```
[ Raw Network Flow CSV in data/raw/ ]
       │
       ▼
[ data_loader.load_dataset / load_dataset_chunks ]
       │──> File validation, 0-byte check, whitespace stripping, duplicate column check
       ▼
[ preprocessing.handle_non_finite_values & handle_duplicates ]
       │──> Converts +inf/-inf to np.nan strictly in numeric columns
       │──> Duplicate auditing and optional removal
       ▼
[ feature_engineering.prepare_feature_matrix ]
       │──> Excludes target and metadata columns (Flow ID, IPs, Timestamps)
       │──> Enforces numeric-only features (no silent coercion)
       │──> Produces reusable FeatureSchema
       ▼
[ preprocessing.encode_binary_target ]
       │──> Maps BENIGN -> 0, Attack -> 1 (rejects missing/unknown labels)
       ▼
[ preprocessing.train_test_split_network_flows / split_data_temporal ]
       │──> Partitions X and y prior to any data-dependent transformations
       ▼
             ├──> X_train, y_train
             │         │
             │         ▼
             │   [ train.build_training_pipeline & train_model ]
             │         │──> SimpleImputer(strategy='median') fitted on X_train ONLY
             │         │──> RandomForestClassifier fitted on X_train
             │         ▼
             │   [ train.save_model_artifact -> models/random_forest_baseline.joblib ]
             │
             └──> X_test, y_test
                       │
                       ▼
                 [ evaluate.evaluate_model & compute_binary_metrics ]
                       │──> Applies fitted pipeline to X_test (uses X_train median)
                       │──> Resolves probability index for attack class
                       │──> Computes Acc, Precision, Recall, F1, FPR, ROC-AUC, PR-AUC, CM
                       ▼
                 [ evaluate.save_evaluation_report -> outputs/reports/*.json ]
```

---

## 2. Pipeline Module Contracts

| Module | Function | Inputs | Outputs | Key Error Handling / Policies |
| :--- | :--- | :--- | :--- | :--- |
| `src/feature_engineering` | `prepare_feature_matrix` | `df`, `target_column`, `drop_metadata` | `(X, y, schema)` | Rejects duplicate columns; excludes metadata; rejects silent coercion |
| `src/feature_engineering` | `FeatureSchema.validate` | `df`, `allow_extra_columns` | `aligned_df` | Verifies presence of all required features; matches exact column order |
| `src/preprocessing` | `encode_binary_target` | `y`, `benign_label`, `attack_labels` | `int64 Series` (0, 1) | Rejects missing/empty labels; rejects unknown labels when whitelist given |
| `src/preprocessing` | `train_test_split_network_flows` | `X`, `y`, `test_size`, `stratify` | `X_tr, X_te, y_tr, y_te` | Rejects datasets with < 2 classes or < 2 samples in minority class |
| `src/preprocessing` | `split_data_temporal` | `df`, `time_column`, `split_ratio` | `X_tr, X_te, y_tr, y_te` | Sorts chronologically; excludes timestamp and target from feature matrix |
| `src/train` | `train_model` | `X_train`, `y_train`, `estimator` | `(pipeline, metadata)` | Packages imputer + estimator; imputer learns median from `X_train` only |
| `src/train` | `save_model_artifact` | `pipeline`, `metadata`, `path`, `overwrite` | `saved_path` | `FileExistsError` if exists and `overwrite=False` |
| `src/evaluate` | `evaluate_model` | `pipeline`, `X_test`, `y_test`, `pos_label` | `metrics_dict` | Resolves `pos_label` probability column dynamically; `zero_division=0.0` |
| `src/predict` | `predict_network_flows` | `model`, `flow_data`, `feature_schema` | `(preds, probs)` | Validates input against schema; checks non-empty input |
| `src/preprocessing` | `sample_dataset` | `df`, `max_samples`, `target_column` | `(df_sampled, meta)` | Controlled reproducible stratified subsampling; never silently drops rows |

---

## 3. Experiment Readiness & Leakage Guarantees

1. **Target-Label Mapping Guarantees**:
   - `DEFAULT_BENIGN_LABELS` (`BENIGN`, `0`, `NORMAL`) map strictly to class `0`.
   - `DEFAULT_RECOGNIZED_ATTACK_LABELS` (canonical CIC-IDS2017 categories: DDoS, PortScan, Bot, Infiltration, Web Attacks, DoS variants, Patator, Heartbleed) map strictly to class `1`.
   - Missing (`NaN`), empty (`""`), and blank whitespace strings are strictly rejected with actionable `ValueError`s.
   - Any unknown label (e.g., unrecognized trojan) raises `ValueError` before training.
   - Target column is strictly purged from feature matrices across all pipelines.

2. **Splitting & Leakage Guarantees**:
   - **Temporal Splitting**: Chronological order is strictly preserved. However, temporal splitting **does not eliminate** host-session or duplicate-flow leakage across split boundaries.
   - **Stratified Random Splitting**: Preserves class ratios, but flows from the same communication session or identical duplicate flows may span train and test partitions.
   - **Cross-day Mixing**: Combining multi-day files into a single random stratified split overstates generalization to unseen network days.
   - **Extreme Minority Classes**: While `sample_dataset()` preserves at least $\min(\text{orig\_count}, 2)$ examples of rare classes to prevent total class omission, preserving 2 examples does not guarantee that both train and test partitions receive that class under subsequent splitting (e.g., `test_size=0.2` on 2 minority rows rounds $2 \times 0.2 = 0.4 \to 0$, leaving the test split with 0 positive samples). When a class is too rare for meaningful evaluation, evaluation reports transparently record single-class test limitations rather than manufacturing artificial balance.

3. **Memory & Computation Readiness**:
   - **Initial Ingestion Constraint**: `load_dataset()` reads the full raw CSV into DataFrame memory before `--max-samples` is applied. `--max-samples` bounds feature matrix conversion, splitting, and Random Forest fitting, but does not reduce the initial CSV file-load footprint.
   - **Estimator Constraints**: `--n-estimators` and `--max-depth` enable bounding the memory footprint of Random Forest tree expansion during model fitting.
   - **No Real Model Training**: Model training was **not executed** during this readiness milestone.

