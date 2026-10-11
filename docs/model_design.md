# AI/ML Model Design & Learning Mechanisms

## 1. Problem Formulation & Classification Paradigm

The initial implementation targets **Binary Intrusion Classification**:
- **Class 0 (Benign)**: Normal background traffic.
- **Class 1 (Attack)**: Any recognized malicious attack vector (DoS, DDoS, PortScan, Brute Force, Web Attacks, etc.).

---

## 2. Supervised Baseline: Random Forest Pipeline

### 2.1 Estimator Architecture
- **Algorithm**: `RandomForestClassifier` (Scikit-learn ensemble of decision trees).
- **Core Parameters**:
  - `n_estimators=100`: Baseline tree ensemble size.
  - `class_weight='balanced'`: Re-weights sample loss inversely proportional to class frequencies to prevent minority attack classes from being ignored.
  - `random_state=42`: Fixed reproducibility seed.
  - `n_jobs=-1`: Parallel tree building across available CPU cores.

### 2.2 Preprocessing Integration (`Pipeline`)
- Missing numerical values are handled via `SimpleImputer(strategy='median')`.
- Imputation statistics are fitted **exclusively on the training split** to prevent data leakage from the test split.
- Numerical feature scaling is intentionally bypassed: Decision trees split features based on monotonic ordering, making them scale-invariant.

### 2.3 Alternative Baseline: Logistic Regression
- For comparative benchmarks, `train_model(..., estimator=LogisticRegression(class_weight='balanced'))` is supported out-of-the-box.

---

## 3. Evaluation & Validation Metrics
Security evaluation prioritizes:
- **False Positive Rate (FPR)**: $FP / (FP + TN)$. High FPR floods SOC analysts with false alarms.
- **Detection Rate (Recall / TPR)**: $TP / (TP + FN)$. Measures true intrusion catch rate.
- **Precision**: $TP / (TP + FP)$. Measures alarm credibility.
- **PR-AUC & ROC-AUC**: Evaluates threshold-independent ranking performance.
- **Confusion Matrix**: Captures complete classification distribution across benign and attack samples.

---

## 4. Probability Semantics & Score Extraction

- **Dynamic Class Indexing**: Estimator `classes_` is dynamically inspected to identify the positive class index corresponding to `1`. The implementation does NOT assume column index 1 corresponds to class `1`.
- **Validation**: Extracted probabilities are verified to be finite and bounded within $[0.0, 1.0]$.
- **Security Interpretation**: Model output scores are decision confidence estimates and should be treated as triage signals rather than calibrated physical incident probabilities.

---

## 5. Current Execution Status
- Tested and verified on synthetic benchmark datasets.
- No real-world training on CIC-IDS2017 has occurred yet due to dataset absence in `data/raw/`.

