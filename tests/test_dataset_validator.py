"""
Automated software tests for raw dataset validation (src/dataset_validator.py).

NOTE: These tests use isolated synthetic fixtures in temporary directories to test
software contracts and memory-aware processing. They are NOT real cybersecurity benchmarks.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.dataset_validator import (
    validate_single_csv,
    validate_dataset_directory,
)
from src.feature_engineering import FeatureSchema
from main import main


def test_validator_missing_directory(tmp_path: Path):
    """Test 1: Validator handles missing data directory gracefully."""
    missing_dir = tmp_path / "does_not_exist"
    report = validate_dataset_directory(missing_dir)
    assert report["status"] == "DIRECTORY_NOT_FOUND"
    assert report["total_files_discovered"] == 0
    assert "Directory not found" in report["error_message"]


def test_validator_empty_directory(tmp_path: Path):
    """Test 2: Validator handles empty data directory gracefully."""
    empty_dir = tmp_path / "empty_raw"
    empty_dir.mkdir()
    report = validate_dataset_directory(empty_dir)
    assert report["status"] == "NO_FILES_FOUND"
    assert report["total_files_discovered"] == 0
    assert "actionable_guidance" in report


def test_validator_one_valid_small_csv(tmp_path: Path):
    """Test 3: Validate a single well-formed synthetic CSV."""
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "Flow Duration": [100.0, 200.0, 300.0, 400.0],
            "Total Packets": [10, 20, 30, 40],
            "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["status"] == "SUCCESS"
    assert res["total_rows"] == 4
    assert res["column_count"] == 3
    assert res["label_frequencies"] == {"BENIGN": 2, "DDoS": 2}
    assert res["label_percentages"] == {"BENIGN": 50.0, "DDoS": 50.0}
    assert res["unknown_labels"] == []


def test_validator_multiple_csv_files(tmp_path: Path):
    """Test 4: Aggregate validation metrics across multiple CSV files."""
    data_dir = tmp_path / "multi_data"
    data_dir.mkdir()

    df1 = pd.DataFrame({"Duration": [10.0, 20.0], "Label": ["BENIGN", "PortScan"]})
    df2 = pd.DataFrame({"Duration": [30.0, 40.0, 50.0], "Label": ["BENIGN", "BENIGN", "DDoS"]})

    df1.to_csv(data_dir / "Monday.csv", index=False)
    df2.to_csv(data_dir / "Tuesday.csv", index=False)

    report = validate_dataset_directory(data_dir)
    assert report["status"] == "SUCCESS"
    assert report["total_files_discovered"] == 2
    agg = report["aggregate_summary"]
    assert agg["total_rows"] == 5
    assert agg["aggregate_label_frequencies"]["BENIGN"] == 3
    assert agg["aggregate_label_frequencies"]["PortScan"] == 1
    assert agg["aggregate_label_frequencies"]["DDoS"] == 1


def test_validator_missing_labels(tmp_path: Path):
    """Test 5: Handle CSV containing missing target labels."""
    csv_file = tmp_path / "missing_labels.csv"
    df = pd.DataFrame(
        {
            "Duration": [10.0, 20.0, 30.0],
            "Label": ["BENIGN", None, "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["missing_values_per_column"]["Label"] == 1
    assert res["total_rows"] == 3
    assert res["label_frequencies"] == {"BENIGN": 1, "DDoS": 1}


def test_validator_unknown_labels(tmp_path: Path):
    """Test 6: Flag unknown or unrecognized attack labels."""
    csv_file = tmp_path / "unknown_label.csv"
    df = pd.DataFrame(
        {
            "Duration": [100.0, 200.0],
            "Label": ["BENIGN", "UNRECOGNIZED_TROJAN_XZ"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["status"] == "VALIDATION_WARNINGS"
    assert "UNRECOGNIZED_TROJAN_XZ" in res["unknown_labels"]


def test_validator_missing_values(tmp_path: Path):
    """Test 7: Accurately quantify missing values per column."""
    csv_file = tmp_path / "missing_features.csv"
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, np.nan],
            "f2": [np.nan, np.nan, np.nan, 4.0],
            "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["missing_values_per_column"]["f1"] == 2
    assert res["missing_values_per_column"]["f2"] == 3
    assert res["total_missing_values"] == 5


def test_validator_infinite_numeric_values(tmp_path: Path):
    """Test 8: Quantify positive and negative infinities in rate columns."""
    csv_file = tmp_path / "inf_features.csv"
    df = pd.DataFrame(
        {
            "Flow Bytes/s": [100.0, np.inf, -np.inf, 200.0],
            "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["positive_inf_counts"]["Flow Bytes/s"] == 1
    assert res["negative_inf_counts"]["Flow Bytes/s"] == 1
    assert res["total_positive_inf"] == 1
    assert res["total_negative_inf"] == 1


def test_validator_all_missing_feature_columns(tmp_path: Path):
    """Test 9: Identify columns that are completely unpopulated (100% NaN)."""
    csv_file = tmp_path / "all_null.csv"
    df = pd.DataFrame(
        {
            "valid_col": [1.0, 2.0, 3.0],
            "all_null_col": [np.nan, np.nan, np.nan],
            "Label": ["BENIGN", "BENIGN", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert "all_null_col" in res["all_missing_columns"]
    assert res["status"] == "VALIDATION_WARNINGS"


def test_validator_malformed_csv_input(tmp_path: Path):
    """Test 10: Handle malformed / corrupted CSV inputs without unhandled crash."""
    corrupted_file = tmp_path / "corrupted.csv"
    with open(corrupted_file, "wb") as f:
        f.write(b"\x00\xff\xfe\x00corrupt,data\n\x80\x81")

    res = validate_single_csv(corrupted_file)
    # Either parses safely with replacement characters or records an ERROR status
    assert res["file_name"] == "corrupted.csv"
    assert res["file_size_bytes"] > 0


def test_validator_timestamp_parsing_failures(tmp_path: Path):
    """Test 11: Identify unparseable timestamps."""
    csv_file = tmp_path / "timestamps.csv"
    df = pd.DataFrame(
        {
            "Timestamp": ["2017-07-07 10:00:00", "07/07/2017 11:30", "NOT_A_VALID_DATE", None],
            "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["timestamp_parsing_failures"] == 1
    assert res["timestamp_missing_count"] == 1


def test_validator_duplicate_columns(tmp_path: Path):
    """Test 12: Identify duplicate column collisions after whitespace normalization."""
    csv_file = tmp_path / "dup_cols.csv"
    with open(csv_file, "w") as f:
        f.write("Port, Port ,Label\n1,2,BENIGN\n")

    res = validate_single_csv(csv_file)
    assert res["status"] == "ERROR"
    assert res["error_type"] == "DuplicateColumnsError"


def test_validator_correct_label_counts_and_percentages(tmp_path: Path):
    """Test 13: Mathematically verify label frequencies and percentages."""
    csv_file = tmp_path / "percentages.csv"
    # 3 Benign (75%), 1 Attack (25%)
    df = pd.DataFrame(
        {
            "feat": [1, 2, 3, 4],
            "Label": ["BENIGN", "BENIGN", "BENIGN", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["label_frequencies"]["BENIGN"] == 3
    assert res["label_frequencies"]["DDoS"] == 1
    assert res["label_percentages"]["BENIGN"] == 75.0
    assert res["label_percentages"]["DDoS"] == 25.0


def test_validator_json_report_generation_and_overwrite(tmp_path: Path):
    """Test 14: Verify JSON report persistence and overwrite protection."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({"feat": [1, 2], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "test.csv", index=False)

    report_path = tmp_path / "report.json"
    report = validate_dataset_directory(data_dir, report_out=report_path, overwrite=False)

    assert report_path.exists()
    with open(report_path) as f:
        loaded = json.load(f)
    assert loaded["status"] == "SUCCESS"
    assert loaded["total_files_discovered"] == 1

    # Overwrite protection test
    with pytest.raises(FileExistsError):
        validate_dataset_directory(data_dir, report_out=report_path, overwrite=False)

    # Overwrite=True succeeds
    validate_dataset_directory(data_dir, report_out=report_path, overwrite=True)


def test_validator_no_model_training_performed(tmp_path: Path):
    """Test 15: Explicitly verify that no model training occurs during validation."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({"feat": [1, 2], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "test.csv", index=False)

    report = validate_dataset_directory(data_dir)
    assert report["model_training_performed"] is False
    assert report["model_evaluation_performed"] is False

    # Verify no .joblib or model artifacts were generated in data_dir
    assert list(data_dir.glob("*.joblib")) == []


def test_validator_chunked_processing_behavior(tmp_path: Path):
    """Test 16: Verify consistent processing across multiple chunks."""
    csv_file = tmp_path / "chunked.csv"
    n_rows = 50
    df = pd.DataFrame(
        {
            "val": list(range(n_rows)),
            "Label": ["BENIGN"] * 25 + ["DDoS"] * 25,
        }
    )
    df.to_csv(csv_file, index=False)

    # Validate with small chunksize=10 (5 chunks)
    res = validate_single_csv(csv_file, chunksize=10)
    assert res["total_rows"] == 50
    assert res["label_frequencies"]["BENIGN"] == 25
    assert res["label_frequencies"]["DDoS"] == 25


def test_cli_validate_data_command(tmp_path: Path):
    """Test 17: Verify CLI 'validate-data' execution."""
    data_dir = tmp_path / "cli_data"
    data_dir.mkdir()
    df = pd.DataFrame({"val": [1, 2], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "flows.csv", index=False)

    report_file = tmp_path / "cli_val_report.json"

    # Positional command style
    exit_code = main(
        ["validate-data", "--data-dir", str(data_dir), "--report-out", str(report_file)]
    )
    assert exit_code == 0
    assert report_file.exists()

    # --mode style with --overwrite
    exit_code_flag = main(
        ["--mode", "validate-data", "--data-dir", str(data_dir), "--report-out", str(report_file), "--overwrite"]
    )
    assert exit_code_flag == 0


def test_validator_duplicates_within_single_chunk(tmp_path: Path):
    """Test 18: Detect duplicate rows within a single chunk."""
    csv_file = tmp_path / "dup_chunk.csv"
    df = pd.DataFrame(
        {
            "val": [1, 2, 2, 3, 3, 3],
            "Label": ["BENIGN", "DDoS", "DDoS", "PortScan", "PortScan", "PortScan"],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file, chunksize=20, duplicate_check="within_file")
    assert res["status"] == "SUCCESS"
    assert res["total_rows"] == 6
    # Rows: [1], [2] (first), [2] (dup 1), [3] (first), [3] (dup 2), [3] (dup 3) -> 3 duplicate rows
    assert res["within_file_duplicate_rows_count"] == 3
    assert res["cross_file_duplicate_rows_count"] == 0
    assert res["duplicate_rows_count"] == 3


def test_validator_duplicates_across_chunks(tmp_path: Path):
    """Test 19: Accurately track duplicate rows that span across chunk boundaries."""
    csv_file = tmp_path / "dup_across_chunks.csv"
    # Create 8 rows with duplicates placed in separate chunks
    df = pd.DataFrame(
        {
            "feat1": [10, 20, 30, 10, 20, 30, 40, 50],
            "feat2": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Label": ["BENIGN", "BENIGN", "DDoS", "BENIGN", "BENIGN", "DDoS", "BENIGN", "DDoS"],
        }
    )
    df.to_csv(csv_file, index=False)

    # Chunk size = 3 -> Chunks: [0..2], [3..5], [6..7]
    # Rows 3, 4, 5 are duplicates of rows 0, 1, 2 across chunk boundaries
    res = validate_single_csv(csv_file, chunksize=3, duplicate_check="within_file")
    assert res["total_rows"] == 8
    assert res["within_file_duplicate_rows_count"] == 3
    assert res["cross_file_duplicate_rows_count"] == 0
    assert res["duplicate_rows_count"] == 3


def test_validator_duplicates_across_files_and_strategies(tmp_path: Path):
    """Test 20: Differentiate within-file, cross-file, and disabled duplicate tracking."""
    data_dir = tmp_path / "multi_file_dups"
    data_dir.mkdir()

    file1 = data_dir / "file1.csv"
    file2 = data_dir / "file2.csv"

    # File 1 has unique rows
    df1 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Label": ["BENIGN", "BENIGN", "DDoS"],
        }
    )
    # File 2 shares rows 2 and 3 with File 1, plus one unique row 4
    df2 = pd.DataFrame(
        {
            "id": [2, 3, 4],
            "Label": ["BENIGN", "DDoS", "PortScan"],
        }
    )
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    # Strategy: cross_file
    rep_cross = validate_dataset_directory(data_dir, duplicate_check="cross_file")
    assert rep_cross["status"] == "SUCCESS"
    agg_cross = rep_cross["aggregate_summary"]
    assert agg_cross["total_within_file_duplicates"] == 0
    assert agg_cross["total_cross_file_duplicates"] == 2
    assert agg_cross["total_duplicate_rows"] == 2

    # Strategy: within_file
    rep_within = validate_dataset_directory(data_dir, duplicate_check="within_file")
    agg_within = rep_within["aggregate_summary"]
    assert agg_within["total_within_file_duplicates"] == 0
    assert agg_within["total_cross_file_duplicates"] == 0
    assert agg_within["total_duplicate_rows"] == 0

    # Strategy: none
    rep_none = validate_dataset_directory(data_dir, duplicate_check="none")
    agg_none = rep_none["aggregate_summary"]
    assert agg_none["total_duplicate_rows"] == 0
    assert agg_none["duplicate_check_strategy"] == "none"


def test_validator_blank_vs_missing_labels(tmp_path: Path):
    """Test 21: Distinguish missing (NaN) labels from blank string labels."""
    csv_file = tmp_path / "blank_labels.csv"
    df = pd.DataFrame(
        {
            "feat": [1, 2, 3, 4, 5, 6],
            "Label": ["BENIGN", None, "   ", "DDoS", "", "  \t  "],
        }
    )
    df.to_csv(csv_file, index=False)

    res = validate_single_csv(csv_file)
    assert res["status"] == "VALIDATION_WARNINGS"
    assert res["total_rows"] == 6
    assert res["missing_labels_count"] == 2  # None and "" serialize as empty CSV fields and parse as NaN
    assert res["blank_labels_count"] == 2    # '   ' and '  \t  ' are non-null whitespace-only strings
    assert res["label_frequencies"] == {"BENIGN": 1, "DDoS": 1}


def test_validator_header_only_zero_rows_csv(tmp_path: Path):
    """Test 22: Handle a CSV file that has header columns but zero data rows."""
    csv_file = tmp_path / "zero_rows.csv"
    with open(csv_file, "w") as f:
        f.write("Flow Duration,Total Packets,Label\n")

    res = validate_single_csv(csv_file)
    assert res["status"] == "VALIDATION_WARNINGS"
    assert res["total_rows"] == 0
    assert res["column_count"] == 3
    assert res["duplicate_rows_count"] == 0


def test_validator_zero_byte_empty_file(tmp_path: Path):
    """Test 23: Handle a completely empty (0-byte) CSV file."""
    csv_file = tmp_path / "zero_byte.csv"
    csv_file.touch()

    res = validate_single_csv(csv_file)
    assert res["status"] == "ERROR"
    assert res["error_type"] == "EmptyFileError"
    assert res["file_size_bytes"] == 0


def test_validator_invalid_chunksize_raises(tmp_path: Path):
    """Test 24: Raise ValueError when chunksize <= 0."""
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1], "Label": ["BENIGN"]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        validate_single_csv(csv_file, chunksize=0)

    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        validate_single_csv(csv_file, chunksize=-10)

    with pytest.raises(ValueError, match="chunksize must be a positive integer"):
        validate_dataset_directory(tmp_path, chunksize=0)


def test_validator_invalid_duplicate_strategy_raises(tmp_path: Path):
    """Test 25: Raise ValueError when an invalid duplicate_check strategy is passed."""
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1], "Label": ["BENIGN"]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Invalid duplicate_check strategy"):
        validate_single_csv(csv_file, duplicate_check="unsupported")

    with pytest.raises(ValueError, match="Invalid duplicate_check strategy"):
        validate_dataset_directory(tmp_path, duplicate_check="unsupported")


def test_cli_duplicate_check_and_invalid_args(tmp_path: Path):
    """Test 26: CLI handles --duplicate-check options and flags invalid chunksize."""
    data_dir = tmp_path / "cli_args_data"
    data_dir.mkdir()
    df = pd.DataFrame({"val": [1, 2], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "flows.csv", index=False)

    # Valid --duplicate-check cross_file
    rep1 = tmp_path / "rep1.json"
    code1 = main(
        ["validate-data", "--data-dir", str(data_dir), "--duplicate-check", "cross_file", "--report-out", str(rep1)]
    )
    assert code1 == 0

    # Valid --duplicate-check none
    rep2 = tmp_path / "rep2.json"
    code2 = main(
        ["validate-data", "--data-dir", str(data_dir), "--duplicate-check", "none", "--report-out", str(rep2)]
    )
    assert code2 == 0

    # Invalid chunksize <= 0
    code_invalid = main(
        ["validate-data", "--data-dir", str(data_dir), "--chunksize", "0"]
    )
    assert code_invalid == 1


def test_validator_report_semantics_and_no_training_guarantee(tmp_path: Path):
    """Test 27: Verify report schema semantics, memory notes, and no training side effects."""
    data_dir = tmp_path / "semantics_data"
    data_dir.mkdir()
    df = pd.DataFrame({"feature1": [1.0, 2.0], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "flows.csv", index=False)

    report_path = tmp_path / "semantics_report.json"
    report = validate_dataset_directory(data_dir, report_out=report_path, duplicate_check="within_file")

    assert "duplicate_hash_guarantee_note" in report["aggregate_summary"]
    assert "memory_complexity_assessment" in report["aggregate_summary"]
    assert report["aggregate_summary"]["duplicate_check_strategy"] == "within_file"
    assert report["model_training_performed"] is False
    assert report["model_evaluation_performed"] is False

    # Confirm JSON on disk matches
    with open(report_path) as f:
        disk_report = json.load(f)
    assert disk_report["aggregate_summary"]["duplicate_check_strategy"] == "within_file"
    assert "64-bit" in disk_report["aggregate_summary"]["duplicate_hash_guarantee_note"]
    assert disk_report["model_training_performed"] is False
