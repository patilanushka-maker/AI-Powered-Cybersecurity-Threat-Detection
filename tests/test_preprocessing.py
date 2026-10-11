"""
Automated unit tests for src/preprocessing.py and feature separation using synthetic fixtures.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import (
    clean_column_headers,
    handle_non_finite_values,
    summarize_missing_and_duplicates,
    handle_duplicates,
    validate_target_labels,
    split_features_and_target,
    encode_binary_target,
    train_test_split_network_flows,
    split_data_temporal,
)
from src.feature_engineering import separate_features_and_target


def test_clean_column_headers_whitespace_and_duplicates():
    """Verify whitespace stripping and duplicate detection on DataFrame headers."""
    df_raw = pd.DataFrame(
        {
            " Destination Port": [80, 443],
            " Flow Duration ": [1200, 3400],
            "Label": ["BENIGN", "DDoS"],
        }
    )
    cleaned = clean_column_headers(df_raw)
    assert list(cleaned.columns) == ["Destination Port", "Flow Duration", "Label"]

    # Duplicates detection
    df_dups = pd.DataFrame({"Flow Bytes/s": [1, 2], " Flow Bytes/s ": [3, 4]})
    with pytest.raises(ValueError):
        clean_column_headers(df_dups)


def test_handle_non_finite_values_converts_inf_and_preserves_valid():
    """
    Verify that +inf and -inf in numerical columns become NaN,
    while valid numbers (including zero and negatives) are strictly preserved.
    """
    df = pd.DataFrame(
        {
            "flow_bytes_s": [150.5, np.inf, -np.inf, 0.0, -42.0],
            "packet_count": [10, 20, 30, 40, 50],
            "protocol_name": ["TCP", "UDP", "HTTP", "infinity", "ICMP"],
        }
    )

    cleaned = handle_non_finite_values(df)

    assert np.isnan(cleaned["flow_bytes_s"].iloc[1])
    assert np.isnan(cleaned["flow_bytes_s"].iloc[2])
    assert cleaned["flow_bytes_s"].iloc[0] == 150.5
    assert cleaned["flow_bytes_s"].iloc[3] == 0.0
    assert cleaned["flow_bytes_s"].iloc[4] == -42.0
    assert list(cleaned["packet_count"]) == [10, 20, 30, 40, 50]
    assert cleaned["protocol_name"].iloc[3] == "infinity"
    assert list(cleaned["protocol_name"]) == ["TCP", "UDP", "HTTP", "infinity", "ICMP"]


def test_summarize_missing_and_duplicates():
    """Verify accurate reporting of missing values per column and duplicate rows."""
    df = pd.DataFrame(
        {
            "feat_1": [1.0, np.nan, 3.0, 1.0],
            "feat_2": [10, 20, 30, 10],
            "Label": ["BENIGN", "DDoS", "BENIGN", "BENIGN"],
        }
    )
    summary = summarize_missing_and_duplicates(df)

    assert summary["total_rows"] == 4
    assert summary["total_columns"] == 3
    assert summary["missing_per_column"]["feat_1"] == 1
    assert summary["missing_per_column"]["feat_2"] == 0
    assert summary["total_missing_cells"] == 1
    assert summary["duplicate_rows_count"] == 1


def test_handle_duplicates_policy():
    """Verify duplicate row inspection vs explicit removal."""
    df = pd.DataFrame(
        {
            "col_a": [1, 2, 1, 3],
            "col_b": ["A", "B", "A", "C"],
        }
    )

    df_kept, count = handle_duplicates(df, remove=False)
    assert count == 1
    assert len(df_kept) == 4

    df_pruned, count_pruned = handle_duplicates(df, remove=True)
    assert count_pruned == 1
    assert len(df_pruned) == 3
    assert list(df_pruned["col_a"]) == [1, 2, 3]


def test_validate_target_labels_missing():
    """Verify that missing/null and empty string target labels are rejected."""
    s_nan = pd.Series(["BENIGN", None, "DDoS"])
    with pytest.raises(ValueError) as exc:
        validate_target_labels(s_nan, target_column="Label")
    assert "missing or null values" in str(exc.value)

    s_empty = pd.Series(["BENIGN", "   ", "DDoS"])
    with pytest.raises(ValueError) as exc:
        validate_target_labels(s_empty, target_column="Label")
    assert "empty string labels" in str(exc.value)


def test_validate_target_labels_unexpected():
    """Verify that unexpected labels outside whitelist are rejected."""
    s = pd.Series(["BENIGN", "DDoS", "Malware_X"])
    with pytest.raises(ValueError) as exc:
        validate_target_labels(s, allowed_labels=["BENIGN", "DDoS", "PortScan"])
    assert "unexpected labels" in str(exc.value)
    assert "Malware_X" in str(exc.value)


def test_split_features_and_target_separation_and_order():
    """Verify features and target separation and order preservation."""
    df = pd.DataFrame(
        {
            "port": [80, 443, 22],
            "duration": [100, 200, 300],
            "Label": ["BENIGN", "DDoS", "PortScan"],
            "bytes_s": [50.0, 75.0, 90.0],
        }
    )

    X, y = split_features_and_target(df, target_column="Label")
    assert "Label" not in X.columns
    assert list(X.columns) == ["port", "duration", "bytes_s"]
    assert len(X) == 3
    assert list(y) == ["BENIGN", "DDoS", "PortScan"]


def test_split_features_and_target_missing_column():
    """Verify KeyError when target column does not exist."""
    df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    with pytest.raises(KeyError) as exc:
        split_features_and_target(df, target_column="Target")
    assert "Target column 'Target' not found" in str(exc.value)


def test_feature_engineering_separate_features_and_target():
    """Verify separate_features_and_target exported in feature_engineering."""
    df = pd.DataFrame(
        {
            "f1": [1, 2],
            "f2": [3, 4],
            "Label": ["BENIGN", "Attack"],
        }
    )
    X, y = separate_features_and_target(df, target_column="Label")
    assert list(X.columns) == ["f1", "f2"]
    assert list(y) == ["BENIGN", "Attack"]


def test_encode_binary_target():
    """Test binary label encoding and rejection of unknown labels."""
    s = pd.Series(["BENIGN", "DDoS", "PortScan", "benign"])
    encoded = encode_binary_target(s, benign_label="BENIGN")
    assert list(encoded) == [0, 1, 1, 0]

    # Whitelist rejection
    s_unknown = pd.Series(["BENIGN", "UnknownExploit"])
    with pytest.raises(ValueError) as exc:
        encode_binary_target(
            s_unknown,
            benign_label="BENIGN",
            attack_labels=["DDoS", "PortScan"],
            reject_unknown=True,
        )
    assert "unexpected label(s)" in str(exc.value).lower()


def test_encode_binary_target_whitespace_and_casing():
    """Test whitespace trimming and case normalization."""
    s = pd.Series(["  benign  ", "  DDOS  ", "Benign", "portscan "])
    encoded = encode_binary_target(s)
    assert list(encoded) == [0, 1, 0, 1]


def test_encode_binary_target_unexpected_benign_like_rejected():
    """Test that unexpected pseudo-benign labels like 'not_an_attack' are rejected."""
    s = pd.Series(["BENIGN", "not_an_attack", "DDoS"])
    with pytest.raises(ValueError) as exc:
        encode_binary_target(s)
    assert "not_an_attack" in str(exc.value)


def test_encode_binary_target_missing_labels_rejected():
    """Test that null and empty labels are strictly rejected."""
    s_null = pd.Series(["BENIGN", None])
    with pytest.raises(ValueError):
        encode_binary_target(s_null)

    s_empty = pd.Series(["BENIGN", "   "])
    with pytest.raises(ValueError):
        encode_binary_target(s_empty)


def test_train_test_split_network_flows_reproducibility():
    """Test reproducible stratified train/test splitting."""
    X = pd.DataFrame({"f1": range(10), "f2": range(10, 20)})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_train1, X_test1, y_train1, y_test1 = train_test_split_network_flows(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    X_train2, X_test2, y_train2, y_test2 = train_test_split_network_flows(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    assert len(X_train1) == 8
    assert len(X_test1) == 2
    assert (y_test1 == 0).sum() == 1
    assert (y_test1 == 1).sum() == 1


def test_train_test_split_unsuitable_datasets():
    """Test failure when dataset is too small or lacks required classes."""
    X_small = pd.DataFrame({"f": [1, 2]})
    y_small = pd.Series([0, 1])
    with pytest.raises(ValueError) as exc:
        train_test_split_network_flows(X_small, y_small)
    assert "too small" in str(exc.value)

    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y_single = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError) as exc:
        train_test_split_network_flows(X, y_single)
    assert "must contain at least 2 distinct classes" in str(exc.value)


def test_split_data_temporal_unsorted_reorders():
    """Test that out-of-order timestamps are sorted chronologically in temporal split."""
    df = pd.DataFrame(
        {
            "Timestamp": [
                "2017-07-07 12:00",
                "2017-07-07 09:00",
                "2017-07-07 11:00",
                "2017-07-07 10:00",
            ],
            "feat": [40, 10, 30, 20],
            "Label": [1, 0, 1, 0],
        }
    )
    X_tr, X_te, y_tr, y_te = split_data_temporal(
        df, time_column="Timestamp", split_ratio=0.75, target_column="Label"
    )
    assert len(X_tr) == 3
    assert len(X_te) == 1
    # Sorted order of feat: 10 (09:00), 20 (10:00), 30 (11:00) in train, 40 (12:00) in test
    assert list(X_tr["feat"]) == [10, 20, 30]
    assert list(X_te["feat"]) == [40]
    assert list(y_te) == [1]


def test_split_data_temporal_invalid_timestamp_raises():
    """Test that unparseable or missing timestamps raise clear ValueError."""
    df = pd.DataFrame(
        {
            "Timestamp": ["2017-07-07 09:00", "invalid_date_str", "2017-07-07 11:00", "2017-07-07 12:00"],
            "feat": [1, 2, 3, 4],
            "Label": [0, 0, 1, 1],
        }
    )
    with pytest.raises(ValueError) as exc:
        split_data_temporal(df, time_column="Timestamp")
    assert "invalid or missing timestamps" in str(exc.value)


def test_split_data_temporal_insufficient_rows():
    """Test error when fewer than 4 rows are provided."""
    df = pd.DataFrame({"Timestamp": ["2017-07-07 09:00", "2017-07-07 10:00"], "Label": [0, 1]})
    with pytest.raises(ValueError) as exc:
        split_data_temporal(df, time_column="Timestamp")
    assert "insufficient rows" in str(exc.value)
