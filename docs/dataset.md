# Dataset Specification & Analysis Guide: CIC-IDS2017

## 1. Dataset Overview & Official Source

- **Dataset Name**: Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017 (CIC-IDS2017)
- **Originating Body**: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
- **Official Dataset Portal**: [UNB CIC-IDS2017 Dataset Page](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Capture Window**: 5 continuous days (Monday, July 3, 2017, 9:00 AM to Friday, July 7, 2017, 5:00 PM).
- **Extraction Mechanism**: Generated using the CICFlowMeter tool from raw PCAP network captures.
- **Current Workspace Status**: **NOT downloaded into `data/raw/`**. All software tests and pipelines are verified via isolated synthetic fixtures. No real CIC-IDS2017 training has occurred.

---

## 2. Dataset Acquisition & Placement Guide

To utilize the genuine benchmark dataset:

1. **Visit the Official UNB Portal**:
   Navigate to [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html) and submit the academic/research access request form.
2. **Download the Pre-extracted Flow CSVs**:
   Download the archive containing the 8 generated network flow CSV files (typically distributed as `GeneratedLabelledFlows.zip` or individual day files).
3. **Target Files**:
   - `Monday-WorkingHours.pcap_ISCX.csv` (Benign background traffic)
   - `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator)
   - `Wednesday-workingHours.pcap_ISCX.csv` (DoS Slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed)
   - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Web Attacks: Brute Force, XSS, SQL Injection)
   - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (Infiltration / PortScan)
   - `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet ARES)
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS LOIC)
4. **Placement in Repository**:
   Extract all `.csv` files directly into:
   ```
   data/raw/
   ```
   Do not commit raw CSV files to Git. The `.gitignore` file automatically excludes `data/raw/*.csv` while preserving `data/raw/.gitkeep`.

---

## 3. Storage, Memory & Compute Considerations

> **⚠️ Resource Estimates Notice**  
> The figures below are estimated resource projections based on UNB documentation and empirical tabular parsing requirements. Operating systems and pandas versions vary in actual overhead.

| Resource Dimension | Estimated Requirement | Notes |
| :--- | :--- | :--- |
| **Compressed Archive** | ~1.0 GB | ZIP archive from UNB portal. |
| **Uncompressed Storage** | ~2.8 GB – 3.1 GB | Total uncompressed disk footprint for all 8 CSV files. |
| **In-Memory Footprint (All Files)** | ~6.0 GB – 12.0 GB RAM | If reading all 8 CSVs into memory simultaneously using pandas. |
| **Recommended Safe Processing** | Chunked streaming (`chunksize=10000` or `50000`) | Enables memory-safe validation and inspection on machines with $\le 8$ GB RAM. |

---

## 4. Dataset Validation Workflow (`validate-data`)

The CLI provides a memory-safe, chunked validator to inspect raw datasets before training:

```bash
# Validate all CSV files in data/raw/ using default within-file duplicate detection
python main.py validate-data --data-dir data/raw --report-out outputs/reports/dataset_validation.json --chunksize 10000

# Perform cross-file duplicate detection across all CSV files
python main.py validate-data --data-dir data/raw --report-out outputs/reports/dataset_validation.json --duplicate-check cross_file

# Run in strict constant-memory mode (duplicate tracking disabled)
python main.py validate-data --data-dir data/raw --report-out outputs/reports/dataset_validation.json --duplicate-check none
```

### Validation Capabilities:
- **Streaming Inspection**: Scans files chunk-by-chunk (`chunksize=10000`, configurable and validated > 0) without loading entire multi-gigabyte datasets into memory.
- **Quality & Non-Finite Profiling**: Computes exact missing value counts, positive infinities (`+inf`), and negative infinities (`-inf`).
- **Label Profiling**: Detects candidate label columns, computes exact frequencies and percentages, reports missing (`NaN`) and blank (`""` / whitespace) labels separately, and flags unknown/unsupported attack labels against canonical whitelists. Unknown labels are never silently mapped to attack or benign.
- **Timestamp Verification**: Identifies timestamp formatting failures (`NaT` count) across mixed datetime representations.
- **Configurable Duplicate Flow Counting**: Supports `within_file`, `cross_file`, and `none` duplicate strategies.
- **Model Schema Audit**: Checks compatibility against `FeatureSchema` (missing features, extra columns, target column presence).
- **Zero Model Side Effects**: **Explicitly does NOT fit transformers, train models, or generate evaluation scores.**

---

## 5. Schema, Duplicate & Leakage Audit Guarantees

### 5.1 Identifier Columns (Target Leakage)
Columns such as `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `Source Port`, and `Destination Port` identify specific host machines, testbed subnets, or experiment schedules. Including them in model features allows estimators to memorize IP addresses or attack hours rather than learning generalized flow dynamics. They are strictly excluded by `DEFAULT_METADATA_COLUMNS`.

### 5.2 Raw Timestamps (Temporal Leakage)
Randomly splitting flows across the entire capture window leaks future traffic dynamics into past training partitions. While `split_data_temporal()` provides chronological ordering, temporal splitting **does not automatically eliminate duplicate-flow or host-session leakage** if flows from the same communication session span split boundaries.

### 5.3 Duplicate Flow Records & Memory Complexity
Identical 5-tuple flow records appearing in both training and test partitions inflate cross-validation accuracy. The dataset validator provides configurable duplicate tracking:

1. **Duplicate Detection Strategies**:
   - `within_file` (Default): Tracks duplicate rows within each individual CSV file using a local 64-bit integer hash set. The hash set is discarded between files, ensuring peak memory does not exceed $O(U_{\text{file}})$ (~30–60 MB RAM per file for ~500k–700k unique rows).
   - `cross_file`: Retains unique 64-bit row hashes in a shared global set across all CSV files. Reports both `within_file_duplicate_rows_count` and `cross_file_duplicate_rows_count`. Peak memory is $O(U_{\text{global}})$ (~80–160 MB RAM for 2.8M unique rows).
   - `none`: Disables duplicate detection entirely. No hash sets are allocated; memory is strictly $O(\text{chunksize} \times \text{columns})$ constant memory.

2. **Probabilistic Nature of 64-Bit Row Hashes**:
   - Duplicate detection uses `pd.util.hash_pandas_object()`, producing 64-bit unsigned integers (`uint64`).
   - **Crucial Mathematical Limitation**: 64-bit hash equality is a probabilistic duplicate estimate, not mathematical proof of identical row values. While birthday-paradox collision odds for $N \approx 3 \times 10^6$ rows with $2^{64}$ buckets are small ($p \approx N^2 / 2^{65} \approx 2.4 \times 10^{-7}$), collisions are theoretically non-zero and could cause slight duplicate overcounting. Exact verification requires storing entire row tuples, which requires gigabytes of RAM.

3. **Distinction Between Chunked Reading and Constant Memory**:
   - Reading CSVs in chunks avoids loading the entire 2.8 GB file into DataFrame memory at once.
   - However, duplicate tracking with Python hash sets grows with the number of unique rows ($O(U)$). True constant-memory processing is achieved when using `--duplicate-check none`.

---

## 6. Usage, Licensing & Terms
- CIC-IDS2017 is provided by the University of New Brunswick for academic and research purposes.
- Raw dataset files must not be committed to source control or redistributed without authorization from UNB.

