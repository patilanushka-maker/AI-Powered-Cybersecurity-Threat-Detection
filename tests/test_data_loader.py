"""
Automated unit tests for src/data_loader.py using synthetic CSV files.
"""

from pathlib import Path
import pytest
import pandas as pd
from src.data_loader import (
    load_dataset,
    load_dataset_chunks,
    normalize_column_names,
    get_dataset_summary,
    list_raw_datasets,
)


def test_load_dataset_valid_csv(tmp_path: Path):
    """Test 1: Successful loading of a valid CSV with correct types."""
    csv_file = tmp_path / "valid_flow.csv"
    csv_file.write_text("Flow Duration,Tot Fwd Pkts,Label\n120,5,BENIGN\n340,12,DDoS\n")

    df = load_dataset(csv_file)
    assert len(df) == 2
    assert list(df.columns) == ["Flow Duration", "Tot Fwd Pkts", "Label"]
    assert df["Flow Duration"].iloc[0] == 120
    assert df["Label"].iloc[1] == "DDoS"


def test_load_dataset_missing_file_raises_error(tmp_path: Path):
    """Test 2: Missing file raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(non_existent)


def test_load_dataset_empty_or_malformed_file(tmp_path: Path):
    """Test 3: Empty (0 bytes) or malformed CSV raises ValueError."""
    # 0 bytes empty file
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        load_dataset(empty_file)

    # File with only newline
    whitespace_file = tmp_path / "whitespace.csv"
    whitespace_file.write_text("\n\n\n")
    with pytest.raises(ValueError):
        load_dataset(whitespace_file)


def test_load_dataset_whitespace_normalization(tmp_path: Path):
    """Test 4: Whitespace normalization in column headers."""
    csv_file = tmp_path / "whitespace_headers.csv"
    csv_file.write_text(" Destination Port ,  Flow Duration  ,  Label \n80,100,BENIGN\n")

    df = load_dataset(csv_file)
    assert list(df.columns) == ["Destination Port", "Flow Duration", "Label"]


def test_load_dataset_detects_duplicate_columns_after_normalization(tmp_path: Path):
    """Test 5: Detection of duplicate column names after normalization."""
    csv_file = tmp_path / "duplicate_headers.csv"
    # 'Flow Rate' and ' Flow Rate ' normalize to the same name
    csv_file.write_text("Flow Rate, Flow Rate ,Label\n10,20,BENIGN\n")

    with pytest.raises(ValueError) as excinfo:
        load_dataset(csv_file)
    assert "Duplicate column names detected after normalization" in str(excinfo.value)


def test_normalize_column_names_utility():
    """Unit test for normalize_column_names utility function."""
    cols = ["  colA ", "colB\t", "colC"]
    assert normalize_column_names(cols) == ["colA", "colB", "colC"]

    # Duplicates test
    with pytest.raises(ValueError):
        normalize_column_names(["colA", " colA "])


def test_load_dataset_chunks_valid(tmp_path: Path):
    """Test chunked reading yielding normalized chunks."""
    csv_file = tmp_path / "chunked.csv"
    csv_file.write_text(" colA , colB \n1,2\n3,4\n5,6\n7,8\n")

    chunks = list(load_dataset_chunks(csv_file, chunksize=2))
    assert len(chunks) == 2
    assert list(chunks[0].columns) == ["colA", "colB"]
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2


def test_get_dataset_summary_accuracy():
    """Test 6: Accurate reporting of row counts, missing values, and duplicates."""
    df = pd.DataFrame(
        {
            "feat_a": [1.0, None, 3.0, 1.0],
            "feat_b": [10.0, 20.0, None, 10.0],
            "Label": ["BENIGN", "DDoS", "BENIGN", "BENIGN"],
        }
    )
    summary = get_dataset_summary(df)
    assert summary["num_rows"] == 4
    assert summary["num_columns"] == 3
    assert summary["columns"] == ["feat_a", "feat_b", "Label"]
    assert summary["missing_values_per_column"]["feat_a"] == 1
    assert summary["missing_values_per_column"]["feat_b"] == 1
    assert summary["missing_values_per_column"]["Label"] == 0
    assert summary["total_missing_values"] == 2
    # Row 0 and Row 3 are identical duplicates
    assert summary["duplicate_rows_count"] == 1


def test_list_raw_datasets(tmp_path: Path):
    """Test list_raw_datasets utility."""
    # When directory has no CSVs
    assert list_raw_datasets(tmp_path) == []

    # When CSVs are added
    (tmp_path / "capture1.csv").write_text("a,b\n1,2\n")
    (tmp_path / "capture2.csv").write_text("c,d\n3,4\n")
    (tmp_path / "notes.txt").write_text("test")

    datasets = list_raw_datasets(tmp_path)
    assert len(datasets) == 2
    assert all(p.suffix == ".csv" for p in datasets)


def test_load_dataset_from_directory_and_provenance(tmp_path: Path):
    """Test loading multiple aligned CSVs from a directory with provenance tracking."""
    (tmp_path / "friday.csv").write_text("Flow Duration,Tot Fwd Pkts,Label\n100,5,BENIGN\n200,10,DDoS\n")
    (tmp_path / "tuesday.csv").write_text("Flow Duration,Tot Fwd Pkts,Label\n300,15,FTP-Patator\n")

    df = load_dataset(tmp_path, provenance_col="capture_source")
    assert len(df) == 3
    assert list(df["capture_source"]) == ["friday.csv", "friday.csv", "tuesday.csv"]
    assert set(df["Label"].unique()) == {"BENIGN", "DDoS", "FTP-Patator"}


def test_load_dataset_directory_schema_mismatch_raises(tmp_path: Path):
    """Test that mismatched columns across CSV files in a directory are rejected."""
    (tmp_path / "valid1.csv").write_text("colA,colB,Label\n1,2,BENIGN\n")
    (tmp_path / "valid2.csv").write_text("colA,different_col,Label\n3,4,DDoS\n")

    with pytest.raises(ValueError) as excinfo:
        load_dataset(tmp_path)
    assert "Schema mismatch across datasets" in str(excinfo.value)


def test_load_dataset_comma_separated_paths(tmp_path: Path):
    """Test loading comma-separated file paths."""
    f1 = tmp_path / "f1.csv"
    f2 = tmp_path / "f2.csv"
    f1.write_text("Flow Duration,Label\n10,BENIGN\n")
    f2.write_text("Flow Duration,Label\n20,SSH-Patator\n")

    df = load_dataset(f"{f1},{f2}")
    assert len(df) == 2
    assert set(df["Label"]) == {"BENIGN", "SSH-Patator"}
