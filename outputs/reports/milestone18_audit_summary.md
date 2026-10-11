# Milestone 18: Independent Reproducibility and Metric Audit Summary

## 1. Audit Overview and Purpose

This audit independently verifies all numerical claims, dataset row counts, confusion matrix calculations, threshold evaluations, and feature collisions reported in Milestone 16 and Milestone 17. 

**Core Rules Followed During Audit**:
* No models were retrained.
* Existing model weights and serialized `.joblib` artifacts were not altered.
* Historical evaluation reports were left untouched.
* The Streamlit dashboard's default model was kept as the multi-attack baseline.
* Wednesday ground-truth labels were **not** used to tune or select models or decision thresholds.

---

## 2. Dataset Verification and Row Counts

All three raw CSV captures were ingested using the project's standard `load_dataset` and `handle_non_finite_values` functions.

| Dataset Capture | Raw Rows Ingested | Clean Rows Processed | Rows Dropped / Rejected | Benign Flows (0) | Attack Flows (1) | Verified Class Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Friday (DDoS)** | 225,745 | 225,745 | 0 (0.00%) | 97,718 | 128,027 | DDoS: 128,027; BENIGN: 97,718 |
| **Tuesday (Patator)** | 445,909 | 445,909 | 0 (0.00%) | 432,074 | 13,835 | BENIGN: 432,074; FTP-Patator: 7,938; SSH-Patator: 5,897 |
| **Wednesday (DoS)** | 692,703 | 692,703 | 0 (0.00%) | 440,031 | 252,672 | BENIGN: 440,031; DoS Hulk: 231,073; DoS GoldenEye: 10,293; DoS slowloris: 5,796; DoS Slowhttptest: 5,499; Heartbleed: 11 |

**Explanation of Zero Dropped Rows**:  
In the project's pipeline, infinite and missing feature values (e.g., in `Flow Bytes/s`) are converted to `np.nan` and imputed using training medians via `SimpleImputer(strategy='median')`. Because rows with non-finite values are imputed rather than discarded, the processed row count exactly matches the raw CSV row count for all datasets.

---

## 3. Recomputed Wednesday Performance Metrics

All five models were evaluated on the full Wednesday dataset ($N=692,703$; 440,031 Benign, 252,672 Attacks). Positive attack probabilities were resolved strictly using the estimator's actual class indices (`classes_`).

### Recomputed Metric Summary Across Decision Thresholds

| Model Name | Threshold | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Accuracy | Precision | Recall | F1-Score | FPR | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Attack RF Baseline** | 0.10 | 30,948 | 2,650 | 437,381 | 221,724 | 67.61% | 92.11% | 12.25% | 0.2162 | 0.602% | 99.40% |
| *(ROC-AUC: 0.8278)* | 0.20 *(Val-Opt)* | 6,554 | 783 | 439,248 | 246,118 | 64.36% | 89.33% | 2.59% | 0.0504 | 0.178% | 99.82% |
| *(PR-AUC: 0.7868)* | 0.30 | 3,433 | 373 | 439,658 | 249,239 | 63.97% | 90.20% | 1.36% | 0.0268 | 0.085% | 99.91% |
| | 0.40 | 2,321 | 239 | 439,792 | 250,351 | 63.82% | 90.66% | 0.92% | 0.0182 | 0.054% | 99.95% |
| | 0.50 *(Default)* | 278 | 74 | 439,957 | 252,394 | 63.55% | 78.98% | 0.11% | 0.0022 | 0.017% | 99.98% |
| **Extra Trees (Balanced)** | 0.10 | 229,328 | 58,229 | 381,802 | 23,344 | 88.22% | 79.75% | 90.76% | 0.8490 | 13.233% | 86.77% |
| *(ROC-AUC: 0.9355)* | 0.20 | 143,756 | 24,825 | 415,206 | 108,916 | 80.69% | 85.27% | 56.89% | 0.6825 | 5.642% | 94.36% |
| *(PR-AUC: 0.8861)* | 0.30 | 95,950 | 16,607 | 423,424 | 156,722 | 74.98% | 85.25% | 37.97% | 0.5254 | 3.774% | 96.23% |
| | 0.40 *(Val-Opt)* | 89,449 | 1,546 | 438,485 | 163,223 | 76.21% | 98.30% | 35.40% | 0.5206 | 0.351% | 99.65% |
| | 0.50 *(Default)* | 40,642 | 257 | 439,774 | 212,030 | 69.35% | 99.37% | 16.08% | 0.2769 | 0.058% | 99.94% |
| **RF Unweighted** | 0.10 | 166,775 | 2,490 | 437,541 | 85,897 | 87.24% | 98.53% | 66.00% | 0.7905 | 0.566% | 99.43% |
| *(ROC-AUC: 0.9208)* | 0.20 | 113,240 | 1,086 | 438,945 | 139,432 | 79.71% | 99.05% | 44.82% | 0.6171 | 0.247% | 99.75% |
| *(PR-AUC: 0.8897)* | 0.30 | 7,226 | 467 | 439,564 | 245,446 | 64.50% | 93.93% | 2.86% | 0.0555 | 0.106% | 99.89% |
| | 0.40 | 200 | 275 | 439,756 | 252,472 | 63.51% | 42.11% | 0.08% | 0.0016 | 0.062% | 99.94% |
| | 0.50 *(Default)* | 92 | 182 | 439,849 | 252,580 | 63.51% | 33.58% | 0.04% | 0.0007 | 0.041% | 99.96% |
| **HistGradientBoosting** | 0.10 | 7,055 | 253 | 439,778 | 245,617 | 64.51% | 96.54% | 2.79% | 0.0543 | 0.057% | 99.94% |
| *(ROC-AUC: 0.4787)* | 0.20 | 6,917 | 179 | 439,852 | 245,755 | 64.50% | 97.48% | 2.74% | 0.0533 | 0.041% | 99.96% |
| *(PR-AUC: 0.3930)* | 0.30 | 6,839 | 164 | 439,867 | 245,833 | 64.49% | 97.66% | 2.71% | 0.0527 | 0.037% | 99.96% |
| | 0.40 | 4,047 | 143 | 439,888 | 248,625 | 64.09% | 96.59% | 1.60% | 0.0315 | 0.032% | 99.97% |
| | 0.50 *(Default)* | 3,176 | 116 | 439,915 | 249,496 | 63.97% | 96.48% | 1.26% | 0.0248 | 0.026% | 99.97% |

---

## 4. Confusion Matrix Arithmetic Audit

For every model and threshold combination, exact programmatic assertions were tested:
1. $\text{TP} + \text{FN} = 252,672$ (Total attack rows) $\to$ **PASSED (100% agreement)**
2. $\text{TN} + \text{FP} = 440,031$ (Total benign rows) $\to$ **PASSED (100% agreement)**
3. $\text{TP} + \text{FP} + \text{TN} + \text{FN} = 692,703$ (Total dataset rows) $\to$ **PASSED (100% agreement)**

All recomputed values match the Milestone 17 reports with zero mathematical discrepancies.

---

## 5. Per-Attack-Family Reconciliation (Extra Trees)

Reconciliation was checked for Extra Trees across all five Wednesday attack categories:

| Attack Category | Total Rows | Detected at $T=0.40$ (TP) | Missed at $T=0.40$ (FN) | Recall at $T=0.40$ | Detected at $T=0.50$ (TP) | Missed at $T=0.50$ (FN) | Recall at $T=0.50$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DoS Hulk** | 231,073 | 88,922 | 142,151 | 38.48% | 40,122 | 190,951 | 17.36% |
| **DoS GoldenEye** | 10,293 | 0 | 10,293 | 0.00% | 0 | 10,293 | 0.00% |
| **DoS slowloris** | 5,796 | 524 | 5,272 | 9.04% | 520 | 5,276 | 8.97% |
| **DoS Slowhttptest** | 5,499 | 3 | 5,496 | 0.05% | 0 | 5,499 | 0.00% |
| **Heartbleed** | 11 | 0 | 11 | 0.00% | 0 | 11 | 0.00% |
| **Total Attack Flows** | **252,672** | **89,449** | **163,223** | **35.40%** | **40,642** | **212,030** | **16.08%** |

* **Row Reconciliation Assertion**: $88,922 + 0 + 524 + 3 + 0 = 89,449$ ($\equiv \text{TP at } T=0.40$). **PASSED**.
* **Row Reconciliation Assertion**: $40,122 + 0 + 520 + 0 + 0 = 40,642$ ($\equiv \text{TP at } T=0.50$). **PASSED**.
* **Total Attack Sum**: $231,073 + 10,293 + 5,796 + 5,499 + 11 = 252,672$. **PASSED**.

---

## 6. Audit of the 4,187 Identical-Feature Claim

To verify the claim that flows labeled `DoS Hulk` in Wednesday share exact identical feature values with flows labeled `BENIGN` in the development captures, an exact value-by-value comparison was conducted across all 78 numerical columns:

1. **Exact Matches Confirmed**: Exactly **4,187 Wednesday DoS Hulk flows** are identical in all 78 numerical features to **38 Development Benign flows**.
2. **Unique Conflicting Vectors**: These matching rows originate from **36 unique 78-feature vectors**.
3. **Reason for Row vs Vector Difference**: In network traffic, high-volume automated attacks repeatedly send identical probing packets (e.g., standard TCP SYN or RST packets with identical window and packet size headers). These 36 specific packet configurations were repeated 4,187 times during the Hulk attack, and identical packets also occurred 38 times during routine benign background traffic in Friday/Tuesday.
4. **Scientific Conclusion**: Based purely on the 78 numerical summary features provided in the dataset, these 4,187 flows are mathematically indistinguishable from normal traffic. This establishes that tabular flow summaries alone have an inherent separability ceiling.

---

## 7. Audit of Decision Threshold Selection

The audit inspected `src/threshold_and_fn_analysis.py` and `outputs/reports/milestone17_threshold_analysis.json`:

* **Validation-Only Selection**: The candidate threshold of $T=0.40$ for Extra Trees and $T=0.20$ for Random Forest was chosen strictly by maximizing the F1-score on the held-out development validation partition ($N=20,000$).
* **No Test Contamination**: Wednesday ground-truth labels were not consulted or utilized when selecting $T=0.40$. The subsequent evaluation on Wednesday represents a genuine out-of-distribution test at a predetermined threshold.

---

## 8. Validation Partition Leakage Findings

An investigation of the development train/validation split procedure revealed an important methodological finding:

* **Sampling Procedure**: 100,000 rows were sampled from Friday + Tuesday using a stratified random sample (`random_state=42`), followed by an 80/20 stratified train/validation split.
* **Flow Leakage Between Partitions**: Because the split was random rather than capture-aware or session-aware, **323 unique feature vectors** appeared in both the training set and the validation set.
* **Quantification**:
  * 1,214 training rows (1.52%) have identical twins in the validation partition.
  * 524 validation rows (2.62%) have identical twins in the training partition.
* **Impact**: Identical flow records spanning train and validation splits artificially inflate in-distribution validation metrics (>99.9% accuracy).
* **Scientific Guidance**: A random train/test split from the same capture environment is **not** an independent test. External capture evaluation (such as Wednesday) is mandatory to discover actual generalization boundaries.

---

## 9. Final Audit Verdict

* **Dataset Count Audit**: **PASS**
* **Wednesday Metrics Recomputation**: **PASS**
* **Confusion Matrix Arithmetic**: **PASS**
* **Per-Family Reconciliation**: **PASS**
* **4,187 Feature Collision Claim**: **PASS (Verified by exact row comparison)**
* **Threshold Selection Independence**: **PASS**
* **Validation Partition Leakage Audit**: **PASS (Documented with warnings on cross-split duplicates)**

**Prototype Notice**: The AI-Powered Cybersecurity Threat Detection System remains an offline intrusion triage prototype. Model scores serve as investigative indicators and do not warrant autonomous network blocking.
