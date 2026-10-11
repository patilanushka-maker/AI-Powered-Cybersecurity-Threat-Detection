# Milestone 21 / 22: Metric Reconciliation and Experimental Validation Summary

> **PROTOTYPE DISCLAIMER**: This system is an **offline CSV-based research prototype** evaluated on CIC-IDS2017 network flow records. It is **NOT** a production intrusion detection system (IDS).

> **MODEL INTEGRITY**: No existing baseline model files have been retrained or overwritten. The default model for app.py remains `models/random_forest_multi_attack.joblib`.

## 1. Executive Summary & Overall Model Comparison

| Model Strategy | Model Type | Threshold ($T$) | Dev Val Recall | Dev Val FPR | Dev Val F1 | Wednesday Recall | Wednesday Precision | Wednesday FPR | Wednesday F1 | Wednesday Alerts |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1 (Frozen)** | RF Multi-Attack | 0.50 | 100.00% | 0.0000% | 1.00000 | **0.11%** | 78.98% | 0.0168% | 0.00220 | 352 |
| **Baseline 2 (Frozen)** | Extra Trees (M16) | 0.50 | 98.02% | 0.0000% | 0.99002 | **16.08%** | 99.37% | 0.0584% | 0.27688 | 40,899 |
| **Baseline 2 (Dev Tuned)** | Extra Trees (M16) | 0.40 | — | — | — | **35.40%** | 98.30% | 0.3513% | 0.52056 | 90,995 |
| **Candidate A (Tree Reg)** | Regularized Extra Trees | 0.50 | 97.98% | 0.0691% | 0.98847 | **38.81%** | 99.29% | 0.1591% | 0.55807 | 98,763 |
| **Candidate B (Anomaly)** | Isolation Forest (Benign-only) | N/A | 2.21% | 0.9610% | 0.04184 | **17.29%** | 90.82% | 1.0031% | 0.29052 | 48,105 |
| **Candidate C (Boosting)** | Regularized HistGradientBoosting | 0.50 | 100.00% | 0.0000% | 1.00000 | **1.34%** | 97.19% | 0.0223% | 0.02648 | 3,489 |

## 2. Complete Confusion Matrix Reconciliation

> **Fixed Class Order**: Index 0 = Negative / Benign (`y=0`), Index 1 = Positive / Attack (`y=1`). Confusion matrix is `[[TN, FP], [FN, TP]]`.

| Model Configuration | Evaluation Split | Threshold | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Predicted Alerts (TP+FP) | Total Rows |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline 1: RF Multi-Attack | Dev Group Val | 0.50 | 15,921 | 0 | 0 | 4,200 | 4,200 | 20,121 |
| Baseline 1: RF Multi-Attack | Wednesday Benchmark | 0.50 | 439,957 | 74 | 252,394 | 278 | 352 | 692,703 |
| Baseline 2: Extra Trees (M16) | Dev Group Val | 0.50 | 15,921 | 0 | 83 | 4,117 | 4,117 | 20,121 |
| Baseline 2: Extra Trees (M16) | Wednesday Benchmark | 0.50 | 439,774 | 257 | 212,030 | 40,642 | 40,899 | 692,703 |
| Baseline 2: Extra Trees (M16) | Wednesday Benchmark | 0.40 | 438,485 | 1,546 | 163,223 | 89,449 | 90,995 | 692,703 |
| Candidate A: Regularized Extra Trees | Dev Group Val | 0.50 | 15,910 | 11 | 85 | 4,115 | 4,126 | 20,121 |
| Candidate A: Regularized Extra Trees | Wednesday Benchmark | 0.50 | 439,331 | 700 | 154,609 | 98,063 | 98,763 | 692,703 |
| Candidate B: Isolation Forest | Dev Group Val | N/A | 15,768 | 153 | 4,107 | 93 | 246 | 20,121 |
| Candidate B: Isolation Forest | Wednesday Benchmark | N/A | 435,617 | 4,414 | 208,981 | 43,691 | 48,105 | 692,703 |
| Candidate C: Regularized HistGradientBoosting | Dev Group Val | 0.50 | 15,921 | 0 | 0 | 4,200 | 4,200 | 20,121 |
| Candidate C: Regularized HistGradientBoosting | Wednesday Benchmark | 0.50 | 439,933 | 98 | 249,281 | 3,391 | 3,489 | 692,703 |

## 3. Candidate A Numerical Reconciliation & Verification of Identities

### Reconciliation of Previously Divergent Candidate A Numbers

In earlier report drafts, three numbers for Candidate A on Wednesday appeared without explicit connection:
1. **98,763 Total Alerts**: Total predicted positives by the model (alerts raised) = $\text{TP} + \text{FP} = 98,063 + 700 = 98,763$.
2. **98,063 True Positives**: Total attack flows correctly detected across all Wednesday attack families = $97,534 \text{ (DoS Hulk)} + 520 \text{ (DoS Slowloris)} + 9 \text{ (DoS GoldenEye)} = 98,063$.
3. **97,534 DoS Hulk True Positives**: True positive detections specifically within the DoS Hulk family (representing 42.21% of 231,073 DoS Hulk flows).

| Metric / Identity | Mathematical Formulation | Evaluated Value | Reconciliation Status |
|---|---|:---:|:---:|
| Evaluation Total Rows | $\text{TN} + \text{FP} + \text{FN} + \text{TP}$ | 439,331 + 700 + 154,609 + 98,063 = **692,703** | $\checkmark$ Matches Wednesday Dataset |
| Total Attack Rows | $\text{TP} + \text{FN}$ | 98,063 + 154,609 = **252,672** | $\checkmark$ Matches Wednesday Attack Support |
| Total Benign Rows | $\text{TN} + \text{FP}$ | 439,331 + 700 = **440,031** | $\checkmark$ Matches Wednesday Benign Support |
| Predicted Alerts | $\text{TP} + \text{FP}$ | 98,063 + 700 = **98,763** | $\checkmark$ Exactly Reconciled |
| Attack Recall | $\text{TP} / (\text{TP} + \text{FN})$ | 98,063 / 252,672 = **38.8104%** | $\checkmark$ Exactly Reconciled |
| False Positive Rate (FPR) | $\text{FP} / (\text{FP} + \text{TN})$ | 700 / 440,031 = **0.1591%** | $\checkmark$ Exactly Reconciled |
| Precision | $\text{TP} / (\text{TP} + \text{FP})$ | 98,063 / 98,763 = **99.2912%** | $\checkmark$ Exactly Reconciled |
| F1-Score | $2 \cdot (\text{Prec} \cdot \text{Rec}) / (\text{Prec} + \text{Rec})$ | **0.55807** | $\checkmark$ Exactly Reconciled |

## 4. Per-Family Attack Coverage & TP Reconciliation

| Attack Family | Support ($N$) | Represented in Training? | Baseline 1 (RF) TP (Rec%) | Baseline 2 (ET) TP (Rec%) | Candidate A (Reg ET) TP (Rec%) | Candidate B (IF) TP (Rec%) | Candidate C (HGB) TP (Rec%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DOS GOLDENEYE** | 10,293 | **No (Absent)** | 0 (0.00%) | 0 (0.00%) | **9 (0.09%)** | 0 (0.00%) | 0 (0.00%) |
| **DOS HULK** | 231,073 | **No (Absent)** | 213 (0.09%) | 40,122 (17.36%) | **97,534 (42.21%)** | 43,441 (18.80%) | 3,282 (1.42%) |
| **DOS SLOWHTTPTEST** | 5,499 | **No (Absent)** | 65 (1.18%) | 0 (0.00%) | **0 (0.00%)** | 0 (0.00%) | 107 (1.95%) |
| **DOS SLOWLORIS** | 5,796 | **No (Absent)** | 0 (0.00%) | 520 (8.97%) | **520 (8.97%)** | 239 (4.12%) | 2 (0.03%) |
| **HEARTBLEED** | 11 | **No (Absent)** | 0 (0.00%) | 0 (0.00%) | **0 (0.00%)** | 11 (100.00%) | 0 (0.00%) |
| **SUM OF ATTACK TPs** | **252,672** | — | **278** | **40,642** | **98,063** | **43,691** | **3,391** |
| **OVERALL ATTACK TP** | **252,672** | — | **278** | **40,642** | **98,063** | **43,691** | **3,391** |
| **RECONCILIATION DELTA** | — | — | **0** (Exact) | **0** (Exact) | **0** (Exact) | **0** (Exact) | **0** (Exact) |

> **Data Pipeline Integrity**: The sum of per-family attack true positives equals the overall attack TP exactly for every model. Furthermore, benign false positives in the family breakdown (700 for Candidate A, 4,414 for Candidate B) match the overall confusion matrix FP count exactly. Both overall and per-family metrics were generated from the exact same underlying prediction arrays.

## 5. Prediction Score Distributions & Empirical Ceiling Analysis

| Flow Class | Row Count | Mean Score | Median Score | Min Score | Max Score | Scores $\ge 0.50$ Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BENIGN** | 440,031 | 0.0405 | 0.0053 | 0.0000 | 0.8863 | 257 |
| **DOS GOLDENEYE** | 10,293 | 0.0942 | 0.1099 | 0.0000 | 0.2900 | 0 |
| **DOS HULK** | 231,073 | 0.3163 | 0.2200 | 0.0033 | 0.6999 | 40,122 |
| **DOS SLOWHTTPTEST** | 5,499 | 0.0422 | 0.0217 | 0.0000 | 0.4545 | 0 |
| **DOS SLOWLORIS** | 5,796 | 0.0910 | 0.0400 | 0.0000 | 0.6157 | 520 |
| **HEARTBLEED** | 11 | 0.0909 | 0.1000 | 0.0300 | 0.1200 | 0 |

> **Scientific Clarification on Score Ceilings**:
- For Baseline Extra Trees evaluated on Wednesday, `DoS GoldenEye` exhibits a maximum prediction score of **0.2900**, `DoS Slowhttptest` has a maximum of **0.4545**, and `Heartbleed` has a maximum of **0.1200**.
- **Important Scientific Note**: These observed score ceilings are an **empirical property of the evaluated model artifact and dataset**, NOT an inherent theoretical limitation of decision tree architectures in general. Decision trees can theoretically partition any continuous feature space; the low scores here result from the model having been trained exclusively on Friday DDoS flood traffic and Tuesday brute-force Patator traffic, whose split thresholds do not activate on these unseen flow patterns.

## 6. Isolation Forest (Candidate B) Verification & Caveats

### Pipeline and Training Provenance Audit
- **Training Invariant**: Fitted strictly on $N=30,000$ benign development flows from Friday and Tuesday (`y == 0`). Zero Wednesday flows were present or used during fitting of the imputer or detector.
- **Prediction Mapping**: Sklearn IsolationForest outputs `+1` for inliers and `-1` for anomalies. The evaluation pipeline correctly mapped `-1 \to 1` (attack / anomaly) and `+1 \to 0` (benign / inlier).
- **Confusion Matrix Reconciliation**:
  * Benign flows evaluated: 440,031 ($\text{TN}=435,617, \text{FP}=4,414$).
  * False Positive Rate: $\text{FPR} = 4,414 / 440,031 = \mathbf{1.0031\%}$.
  * Earlier report draft text inadvertently quoted an uncommitted exploratory run figure of '1.18% (5,193 false alerts)'. This has been reconciled: the canonical pipeline outputs $\text{FP}=4,414$ and $\text{FPR}=1.0031\%$.

### Heartbleed Preliminary Evidence Caveat
> **CRITICAL SCIENTIFIC CAVEAT**: While Candidate B detected **11/11 (100%) Heartbleed flows** as statistical payload/variance anomalies, **11 flows constitute preliminary evidence only**. A sample size of $N=11$ is insufficient to establish reliable detection or generalizability in production. Furthermore, detecting these 11 flows came at the cost of 4,414 false-positive alerts on normal traffic, which is prohibitive for high-throughput enterprise networks.

## 7. Experiment Fairness & Baseline Integrity Audit

| Model Artifact Path | SHA-256 Checksum | Invariant Status |
|---|---|:---:|
| `models/random_forest_multi_attack.joblib` | `486d6c75f00a163204cb993d0a4e9e2688c1b22ef58cd58e2e53e418d7fcd35f` | **PRESERVED UNCHANGED** |
| `models/model_extra_trees_balanced_m16.joblib` | `c60cef05d9ad7751f013e13d5c297dc2e9b1191ec9aaf1af887cb16227c7f60b` | **PRESERVED UNCHANGED** |

- **Default Model Active in `app.py`**: `models/random_forest_multi_attack.joblib remains active default in app.py`.
- **Data Leakage Prevention**: Verified via group-aware splitting on development data (zero overlapping feature vectors between train and validation).
- **Strict Threshold Discipline**: All thresholds ($T=0.50$, $T=0.40$) were selected on development validation only. Wednesday ground-truth labels were strictly NOT used for threshold tuning.

## 8. Scientific Conclusions & Limitations

### Demonstrated Generalization Findings
- 1. Candidate A (Regularized Extra Trees) significantly improved cross-capture attack recall on Wednesday from 16.08% to 38.81% at standard threshold 0.50 (+141% recall improvement), detecting 98,063 attacks (97,534 Hulk, 520 Slowloris, 9 GoldenEye) with 99.29% precision across 98,763 alerts (700 false alarms).
- 2. Candidate B (Isolation Forest) demonstrated that unsupervised novelty modeling on benign baselines can detect zero-day payload outliers like Heartbleed (11/11 detected, preliminary evidence only on 11 flows), which supervised tree models completely missed.

### Persistent Limitations
- 1. Low-rate HTTP application attacks (DoS GoldenEye 0.09%, DoS Slowhttptest 0.00%, DoS Slowloris 8.97%) remain severely undetected across all supervised classifiers because their summary 4-tuple statistics blend into normal HTTP sessions.
- 2. Candidate B anomaly detection incurs an elevated false-positive rate (1.003%, 4,414 false alerts on 440,031 Benign flows), producing 48,105 total alerts which is burdensome for high-throughput networks.
- 3. Candidate C (Gradient Boosting) suffers severe out-of-distribution regression (1.34% recall), proving that high validation performance does not translate to cross-capture resilience.

### Recommendation on Default Model: Keep 'models/random_forest_multi_attack.joblib' as the default production fallback model. Candidate A is validated as a superior experimental generalization model and should be documented alongside Extra Trees M16.
