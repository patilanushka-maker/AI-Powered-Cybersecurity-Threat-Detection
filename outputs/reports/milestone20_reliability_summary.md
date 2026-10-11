# Milestone 20: Generalization-Focused Evaluation and Model Reliability Audit

> **PROTOTYPE DISCLAIMER**: This system is an **offline CSV-based research prototype** evaluated on CIC-IDS2017 network flow records. It is **NOT** a production intrusion detection system (IDS) and must not be described as production-ready.

## 1. Phase 1: Implementation Correctness & Pipeline Audit

| Audit Check | Status | Verification Detail |
|---|:---:|---|
| Feature Column Alignment | **PASS** | All 78 features ordered identically between schema and model. |
| Dynamic Class Ordering | **PASS** | Positive class index dynamically resolved from estimator `classes_` attribute. |
| Positive-Class Probability Bounds | **PASS** | Verified finite and strictly bounded within $[0.0, 1.0]$. |
| Exact Duplicate Detection | **PASS** | Full feature-vector equality confirmed; 0 conflicting label groups detected. |
| Missing & Infinite Value Handling | **PASS** | Non-finite values safely transformed; median imputation fitted exclusively on training rows. |
| Label Leakage into Features | **PASS** | Target `Label` and metadata identifiers strictly excluded prior to feature matrix construction. |

## 2. Phase 2 & 3: Separated Generalization Evaluations

To avoid misleading aggregation, in-distribution validation, duplicate sensitivity, group-aware partitioning, and cross-capture transfer are reported separately.

### In-Distribution Validation Spectrum (Split A, B, G)

| Split Type | Model | Samples | Accuracy | Precision | Recall | F1 Score | FPR | Boundary Leakage |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Split A (Random Stratified)** | RF Multi-Attack | 20,000 | 0.99975 | 1.00000 | 0.99882 | 0.99941 | 0.000000 | 523 rows (2.61%) |
| **Split A (Random Stratified)** | Extra Trees | 20,000 | 0.99600 | 0.99976 | 0.98130 | 0.99044 | 0.000063 | 523 rows (2.61%) |
| **Split B (Duplicate-Purged)** | RF Multi-Attack | 19,477 | 0.99974 | 1.00000 | 0.99878 | 0.99939 | 0.000000 | **0 rows (Purged)** |
| **Split B (Duplicate-Purged)** | Extra Trees | 19,477 | 0.99933 | 0.99976 | 0.99708 | 0.99842 | 0.000065 | **0 rows (Purged)** |
| **Split G (Group-Aware)** | RF Multi-Attack | 20,121 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.000000 | **0 groups overlap** |
| **Split G (Group-Aware)** | Extra Trees | 20,121 | 0.99587 | 1.00000 | 0.98024 | 0.99002 | 0.000000 | **0 groups overlap** |

> **Finding on In-Distribution Leakage**: Purging boundary duplicates (Split B) or enforcing strict group-aware splitting (Split G) does not degrade in-distribution validation performance. High validation accuracy is an artifact of sampling from the same capture environment, not duplicate vector memorization.

### Cross-Capture Transfer Benchmarks (Split C, D, E)

> **CRITICAL CONFLATION NOTE**: Friday contains only DDoS flood traffic, while Tuesday contains only FTP-Patator and SSH-Patator brute-force traffic. Transfers between Friday and Tuesday **confound capture shift with complete attack-family shift** and do **NOT** isolate the causal effect of network capture shift.

| Transfer Setting | Evaluated Model | Evaluation Rows | Accuracy | Precision | Recall | F1 Score | FPR | Attack Recall / Total Attack |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Friday $\to$ Tuesday | RF Baseline (Friday DDoS) | 445,909 | 0.96894 | 0.00000 | **0.00000** | 0.00000 | 0.000035 | 0 / 13,835 (0.0%) |
| Tuesday $\to$ Friday | RF (Tuesday Patator) | 225,745 | 0.43287 | 0.00000 | **0.00000** | 0.00000 | 0.000000 | 0 / 128,027 (0.0%) |
| Dev $\to$ Wednesday (*benchmark*) | RF Multi-Attack ($T=0.50$) | 692,703 | 0.63553 | 0.78977 | **0.00110** | 0.00220 | 0.000168 | 278 / 252,672 (0.11%) |
| Dev $\to$ Wednesday (*benchmark*) | Extra Trees ($T=0.50$) | 692,703 | 0.69354 | 0.99372 | **0.16085** | 0.27688 | 0.000584 | 40,642 / 252,672 (16.08%) |
| Dev $\to$ Wednesday (*benchmark*) | Extra Trees ($T=0.40$) | 692,703 | 0.76214 | 0.98301 | **0.35401** | 0.52056 | 0.003513 | 89,449 / 252,672 (35.40%) |

## 3. Phase 4: Attack-Family Coverage & Explicit Warnings

| Attack Family | Capture Source | Represented in Training? | Support | ET (T=0.50) Recall | ET (T=0.40) Recall | Coverage Status / Explicit Warning |
|---|---|:---:|:---:|:---:|:---:|---|
| **DOS GOLDENEYE** | Wednesday | **ABSENT (Unseen)** | 10,293 | 0.00% | 0.00% | EXPLICIT WARNING: Family 'DOS GOLDENEYE' was completely ABSENT from training. Performance reflects zero-shot transfer, NOT ordinary in-distribution generalization. |
| **DOS HULK** | Wednesday | **ABSENT (Unseen)** | 231,073 | 17.36% | 38.48% | EXPLICIT WARNING: Family 'DOS HULK' was completely ABSENT from training. Performance reflects zero-shot transfer, NOT ordinary in-distribution generalization. |
| **DOS SLOWHTTPTEST** | Wednesday | **ABSENT (Unseen)** | 5,499 | 0.00% | 0.05% | EXPLICIT WARNING: Family 'DOS SLOWHTTPTEST' was completely ABSENT from training. Performance reflects zero-shot transfer, NOT ordinary in-distribution generalization. |
| **DOS SLOWLORIS** | Wednesday | **ABSENT (Unseen)** | 5,796 | 8.97% | 9.04% | EXPLICIT WARNING: Family 'DOS SLOWLORIS' was completely ABSENT from training. Performance reflects zero-shot transfer, NOT ordinary in-distribution generalization. |
| **HEARTBLEED** | Wednesday | **ABSENT (Unseen)** | 11 | 0.00% | 0.00% | EXPLICIT WARNING: Family 'HEARTBLEED' was completely ABSENT from training. Performance reflects zero-shot transfer, NOT ordinary in-distribution generalization. |

## 4. Phase 5: Threshold Sensitivity Spectrum

Comparing threshold behavior across development validation and the Wednesday benchmark confirms non-stationarity:

### Extra Trees Threshold Spectrum

| Threshold | Dev Val Recall | Dev Val FPR | Dev Predicted Alerts | Wednesday Recall | Wednesday FPR | Wednesday Alerts |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.20** | 99.98% | 2.3327% | 4,591 | 56.89% | 5.6416% | 168,581 |
| **0.30** | 99.88% | 1.4642% | 4,450 | 37.97% | 3.7741% | 112,557 |
| **0.40** | 99.83% | 0.1902% | 4,247 | 35.40% | 0.3513% | 90,995 |
| **0.50** | 98.13% | 0.0063% | 4,146 | 16.08% | 0.0584% | 40,899 |
| **0.60** | 98.08% | 0.0000% | 4,143 | 6.71% | 0.0275% | 17,078 |
| **0.70** | 98.08% | 0.0000% | 4,143 | 0.00% | 0.0225% | 99 |

> **Threshold Non-Stationarity Finding**: Lowering the threshold to $0.20$ increases Wednesday attack recall to 56.89%, but increases the false-positive rate to 5.64% (generating 168,581 alerts). Operating thresholds tuned on development captures cannot be safely deployed across distinct environments without real-time threshold calibration.

## 5. Scientific Distinctions and Conclusions

### Verified Findings
- 1. Random row-level validation (Split A) produces near-perfect scores (>0.999 F1) that do not reflect cross-capture capability.
- 2. Purging boundary-crossing duplicate rows (Split B) or using group-aware splitting (Split G) barely degrades in-distribution validation metrics (>0.99 F1).
- 3. Transfer between Friday and Tuesday produces 0% attack recall in both directions because the training captures completely lack the target attack families.
- 4. Low-rate DoS attacks (GoldenEye, Slowloris, Slowhttptest) and TLS exploit Heartbleed cannot be detected by summary 4-tuple statistics without prior training.
- 5. Thresholds selected on development validation data do not maintain the same false-positive rate on independent captures.

### Hypotheses Requiring Additional Investigation
- 1. Application-layer HTTP header parsing or connection-state metrics may distinguish slow-rate HTTP DoS without full packet payload inspection.
- 2. Deep packet inspection (DPI) of TLS records may be required to detect single-packet exploits like Heartbleed.
- 3. Host-level multi-flow temporal graph clustering may detect distributed brute-force attacks across capture shifts.

### Known Limitations
- 1. Evaluation is restricted to static offline CIC-IDS2017 CSV flow records.
- 2. The dataset contains synthetic background traffic that does not represent live campus or enterprise network dynamics.
- 3. Cross-capture transfer cannot isolate the pure effect of capture environment shift because attack families are partitioned across different days.

### Unsupported Claims Explicitly Rejected
- REJECTED CLAIM 1: 'High validation accuracy indicates the system is ready for production IDS deployment.' (Offline CSV prototype only).
- REJECTED CLAIM 2: 'Deduplication guarantees an evaluation is more representative of real traffic.' (Deduplication alters natural flow arrival rates).
- REJECTED CLAIM 3: 'Friday-to-Tuesday transfer isolates network capture shift.' (Confounded with complete attack-family shift).

## 6. Formal Audit Reconciliation (Verdict: VERIFIED WITH CAVEATS)

> **Verdict Rationale**: All implementation checks, mathematical reconciliations, and group-aware split verifications passed. The audit confirms that offline in-distribution metrics (>0.99 F1) do not generalize across capture boundaries (0% attack recall on transfer without target attack families in training).

### Model Artifact SHA-256 Checksums (Integrity Invariant)

| Model Artifact Path | SHA-256 Checksum | Invariant Status |
|---|---|:---:|
| `models/random_forest_multi_attack.joblib` | `486d6c75f00a163204cb993d0a4e9e2688c1b22ef58cd58e2e53e418d7fcd35f` | **PRESERVED** |
| `models/model_extra_trees_balanced_m16.joblib` | `c60cef05d9ad7751f013e13d5c297dc2e9b1191ec9aaf1af887cb16227c7f60b` | **PRESERVED** |
| `models/random_forest_baseline.joblib` | `e6e6d785ab068381befcfbc914402be9253cec6789ac55c820952b670a5bc3b3` | **PRESERVED** |

### Checks Passed
- [x] Sampling and split reproducibility verified (100,000 dev rows, random_state=42)
- [x] Group-aware split zero leakage verified (97,919 unique groups, 0 overlapping groups across partitions)
- [x] Exact duplicate detection verified (full-column equality confirmed, 0 conflicting label groups in dev data)
- [x] Model class ordering dynamic resolution verified (classes_ attribute inspected dynamically)
- [x] Positive-class probability extraction verified (finite, bounded within [0.0, 1.0])
- [x] Threshold comparison consistency verified (vectorized inequality on documented thresholds)
- [x] Confusion matrix reconciliation verified (sum matches evaluated sample counts for all splits)
- [x] Attack family reconciliation verified (per-family counts reconcile to capture ground truth)
- [x] Artifact integrity verified (model checksums identical before and after audit)
- [x] Default model invariance verified (models/random_forest_multi_attack.joblib unchanged in app.py)

### Checks Failed
- None (0 checks failed).

### Checks Not Independently Verified
- [!] Live network packet capture and active inline blocking (out of scope for offline research prototype)
- [!] Real-world campus/enterprise generalization outside CIC-IDS2017 dataset

### Discrepancies Found & Explanations
- No numerical discrepancies found across recomputed metrics.

### Reasons for Metric Differences Across Milestone Reports
- Differences across Milestones 16-20 reflect distinct evaluation partitions: M16/M17/M18 evaluated Split A (random 20,000 held-out), M19 evaluated Split B (purged 19,477 held-out) and cross-capture, while M20 evaluated Split G (group-aware 20,121 held-out) and an expanded threshold spectrum (0.20 to 0.70).
- All metric calculations reconcile strictly within each designated split.
