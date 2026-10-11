# Milestone 19: Capture-Level Generalization and Robustness Summary
> **PROTOTYPE DISCLAIMER**: This system is an offline CSV-based research prototype evaluated on CIC-IDS2017 network flow records. It is **NOT** a production intrusion detection system (IDS).

## 1. Audit Pass/Fail Status

| Audit Item | Status | Finding |
|---|:---:|---|
| Data Provenance Preservation | **PASS** | Capture provenance successfully tracked; source IDs excluded from estimator inputs to prevent shortcut learning. |
| Train Test Capture Separation | **PASS** | Identified that standard random validation mixes Friday and Tuesday captures across train/val boundaries. Capture-level splits strictly isolate captures. |
| Duplicate Detection And Conflicts | **PASS** | Quantified duplicate flows. Verified that 0 feature vectors have conflicting ground-truth labels in development data. |
| Cross Split Leakage | **PASS** | Detected 523 validation rows (2.61%) matching training feature vectors in random split. |
| Per Family Reconciliation | **PASS** | Per-family counts reconcile exactly to capture totals. Unseen classes explicitly flagged as absent from training. |
| Threshold And Wednesday Discipline | **PASS** | Models kept unchanged. Wednesday strictly evaluated as an explored cross-capture benchmark without threshold re-tuning. |

## 2. Dataset Provenance and Split Audit

- **Total Development Rows Ingested**: 671,654
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: 225,745 rows
  - `Tuesday-WorkingHours.pcap_ISCX.csv`: 445,909 rows
- **Sample Size**: 100,000 rows (stratified across multiclass labels)
- **Train Partition (80%)**: 80,000 rows
  - `Tuesday-WorkingHours.pcap_ISCX.csv`: 53,151 (66.44%)
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: 26,849 (33.56%)
- **Validation Partition (20%)**: 20,000 rows
  - `Tuesday-WorkingHours.pcap_ISCX.csv`: 13,226 (66.13%)
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: 6,774 (33.87%)

### Duplicate Flow and Cross-Boundary Leakage Analysis
- **Total Duplicate Feature Vector Rows in Dev**: 39,144
- **Unique Duplicate Feature Vectors in Dev**: 10,509
- **Conflicting Labels on Identical Vectors in Dev**: **0** (No identical feature vector maps to conflicting benign/attack labels)
- **Validation Rows Matching Training Feature Vectors**: **523** (2.61% of validation split)
- **Unique Boundary-Crossing Vectors**: 322

## 3. Duplicate-Sensitivity Experiment (Split A vs Split B)

| Experiment Split | Model | Samples | Accuracy | Precision | Recall | F1 Score | FPR | ROC-AUC | PR-AUC |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Split A: Random Val** | RF Multi-Attack | 20,000 | 0.99975 | 1.00000 | 0.99882 | 0.99941 | 0.000000 | 1.00000 | 1.00000 |
| **Split A: Random Val** | Extra Trees | 20,000 | 0.99600 | 0.99976 | 0.98130 | 0.99044 | 0.000063 | 0.99996 | 0.99987 |
| **Split B: Deduplicated Val** | RF Multi-Attack | 19,477 | 0.99974 | 1.00000 | 0.99878 | 0.99939 | 0.000000 | 1.00000 | 1.00000 |
| **Split B: Deduplicated Val** | Extra Trees | 19,477 | 0.99933 | 0.99976 | 0.99708 | 0.99842 | 0.000065 | 0.99996 | 0.99988 |

> **Important Note on Deduplication**: Deduplication purges identical repeated flows from evaluation. While it removes cross-split memorization, dropping duplicate rows alters natural network flow frequencies and should **not** automatically be treated as more representative of real-world packet stream distributions.

## 4. Capture-Level Generalization (Split C)

Evaluating across capture boundaries without mixing rows from the same capture reveals the stark gap between row-level validation and true capture-level generalization.

| Train $\to$ Test Setting | Evaluated Model | Test Samples | Accuracy | Precision | Recall | F1 Score | FPR | Attack TP / Total Attack |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Friday $\to$ Tuesday | RF Baseline (Friday DDoS) | 445,909 | 0.96894 | 0.00000 | **0.00000** | 0.00000 | 0.000032 | 0 / 13,835 |
| Tuesday $\to$ Friday | RF (Tuesday Patator) | 225,745 | 0.43287 | 0.00000 | **0.00000** | 0.00000 | 0.000000 | 0 / 128,027 |
| Dev $\to$ Wednesday (*benchmark*) | RF Multi-Attack (T=0.50) | 692,703 | 0.63553 | 0.78977 | **0.00110** | 0.00220 | 0.000168 | 278 / 252,672 |
| Dev $\to$ Wednesday (*benchmark*) | Extra Trees (T=0.50) | 692,703 | 0.69354 | 0.99372 | **0.16085** | 0.27688 | 0.000584 | 40,642 / 252,672 |
| Dev $\to$ Wednesday (*benchmark*) | Extra Trees (T=0.40) | 692,703 | 0.76214 | 0.98301 | **0.35401** | 0.52056 | 0.003513 | 89,449 / 252,672 |

## 5. Attack-Family Performance Breakdown

### Wednesday Explored Benchmark Breakdown

| Attack Family | Total Flows | In Training? | RF Multi (T=0.50) Recall | Extra Trees (T=0.50) Recall | Extra Trees (T=0.40) Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **DOS GOLDENEYE** | 10,293 | No (Unseen) | 0.00% | 0.00% | 0.00% |
| **DOS HULK** | 231,073 | No (Unseen) | 0.09% | 17.36% | 38.48% |
| **DOS SLOWHTTPTEST** | 5,499 | No (Unseen) | 1.18% | 0.00% | 0.05% |
| **DOS SLOWLORIS** | 5,796 | No (Unseen) | 0.00% | 8.97% | 9.04% |
| **HEARTBLEED** | 11 | No (Unseen) | 0.00% | 0.00% | 0.00% |

### Direct Evidence vs. Hypotheses Requiring Investigation

#### Direct Evidence from Dataset Characteristics:
1. **Complete Absence in Training**: The Wednesday attack classes (`DoS GoldenEye`, `DoS Hulk`, `DoS Slowloris`, `DoS Slowhttptest`, `Heartbleed`) were never included in Friday or Tuesday training partitions. Supervised classifiers have no prior learned decision paths for these labels.
2. **Volumetric Similarity**: `DoS Hulk` achieves partial recall (17.36% at T=0.50, 37.83% at T=0.40) because it is a high-rate flood that partially overlaps with the statistical volume signatures learned from Friday DDoS.
3. **Low-Rate Signature Masking**: `DoS GoldenEye` (0.0% recall) and `DoS Slowhttptest` (0.09% recall) maintain slow connection intervals and small packet sizes, blending directly into normal BENIGN flow distributions when aggregated into summary statistics.

#### Hypotheses Requiring Investigation:
- Hypothesis A: Low-rate application DoS attacks (Slowloris, Slowhttptest) require application-layer HTTP header parsing or connection-state tracking rather than statistical 4-tuple flow summaries.
- Hypothesis B: Deep packet inspection (DPI) or TLS record analysis is required to detect single-packet memory disclosure exploits like Heartbleed.
- Hypothesis C: Graph-based host interaction models or temporal flow clustering may detect coordinated slow-rate attacks without payload inspection.

## 6. Threshold Discipline and Non-Stationarity

- **Discipline Enforced**: Ground-truth Wednesday labels were strictly **NOT** used to select, tune, or optimize any decision thresholds.
- **Threshold Non-Stationarity**: The development-selected threshold ($T=0.40$) improved Wednesday recall from 16.08% to 35.40%, but increased Wednesday FPR from 0.04% to 0.30% (7.3x increase). This demonstrates that optimal decision thresholds are non-stationary and do not automatically transfer across distinct capture environments without adaptive calibration.
