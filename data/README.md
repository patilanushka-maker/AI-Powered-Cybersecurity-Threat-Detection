# Dataset Directory Guide

This directory manages all local data assets for the **AI-Powered Cybersecurity Threat Detection System**.

---

## 1. Intended Dataset Source
- **Benchmark**: **CIC-IDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset)
- **Official Dataset Page**: [UNB CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Status in Workspace**: Not yet downloaded. Automated tests validate pipeline contracts using synthetic data fixtures.

---

## 2. Directory Layout & Storage Policy

```
data/
├── raw/         # Place original downloaded CSV captures here (e.g., Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv)
├── processed/   # Transformed, scaled, and split matrices ready for ML modeling
└── README.md    # Data usage instructions (this file)
```

- **Location for Raw Captures**: Store uncompressed CSV files in `data/raw/`.
- **Git Exclusions**: All CSV, PCAP, Parquet, and archive files are ignored by `.gitignore`.
- Raw files must remain immutable; preprocessing output is directed to `data/processed/`.

---

## 3. Data Loader Usage (`src/data_loader.py`)

The loader provides safe CSV ingestion with whitespace stripping, duplicate column detection, and chunked streaming:

```python
from pathlib import Path
from src.data_loader import load_dataset, load_dataset_chunks, get_dataset_summary

# 1. Full dataset load
df = load_dataset(Path("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))

# 2. Chunked loading for large CSV files
for chunk in load_dataset_chunks("data/raw/large_capture.csv", chunksize=50000):
    # Process chunk
    pass

# 3. Operational summary report
summary = get_dataset_summary(df)
print(f"Total rows: {summary['num_rows']}, Missing cells: {summary['total_missing_values']}")
```

---

## 4. Ingestion & Preprocessing Behavior

1. **Column Normalization**:
   - Header names are cast to strings and stripped of whitespace via `normalize_column_names()`.
   - If two columns collide after stripping (e.g. `'Port'` and `' Port '`), a `ValueError` is raised.
2. **Infinite and Missing Values**:
   - In numerical columns, `+inf` and `-inf` are converted to `np.nan` via `handle_non_finite_values()`.
   - String/object columns and valid numbers are preserved without corruption.
3. **Duplicate Row Handling**:
   - `handle_duplicates(df, remove=False)` reports duplicate count without dropping rows.
   - Calling with `remove=True` explicitly drops duplicates and returns `(cleaned_df, duplicate_count)`.
4. **Label Validation & Feature Splitting**:
   - `split_features_and_target(df, target_column="Label", reject_missing=True)` separates features `X` and target `y`.
   - Ensures `target_column not in X.columns`, preserves original column order, and rejects missing or unauthorized labels.

---

## 5. Pending Implementation (Future Milestones)
- Automated imputation strategies on train-split numeric features.
- Feature scaling (RobustScaler / StandardScaler) fitted exclusively on train split.
- Zero-variance and high-correlation feature pruning.
