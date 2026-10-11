"""
Automated unit tests for src/feature_engineering.py.
"""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import (
    FeatureSchema,
    prepare_feature_matrix,
    separate_features_and_target,
)


def test_feature_target_separation_and_exclusion():
    """Test 1 & 2: Feature/target separation and strict target exclusion."""
    df = pd.DataFrame(
        {
            "packet_rate": [10.5, 20.1, 5.2],
            "byte_count": [1000, 2500, 400],
            "Label": ["BENIGN", "DDoS", "BENIGN"],
        }
    )
    X, y, schema = prepare_feature_matrix(df, target_column="Label")

    assert "Label" not in X.columns
    assert list(X.columns) == ["packet_rate", "byte_count"]
    assert list(y) == ["BENIGN", "DDoS", "BENIGN"]
    assert schema.target_column == "Label"


def test_feature_order_consistency():
    """Test 3: Feature-order consistency when explicit schema is provided."""
    df = pd.DataFrame(
        {
            "feat_c": [3, 4],
            "feat_a": [1, 2],
            "feat_b": [5, 6],
            "Label": [0, 1],
        }
    )
    desired_order = ["feat_a", "feat_b", "feat_c"]
    X, y, schema = prepare_feature_matrix(
        df, target_column="Label", feature_columns=desired_order
    )

    assert list(X.columns) == desired_order
    assert list(schema.feature_names) == desired_order


def test_missing_required_feature_errors():
    """Test 4: Error raised when required features are missing."""
    df = pd.DataFrame({"feat_a": [1, 2], "Label": [0, 1]})

    with pytest.raises(KeyError) as exc:
        prepare_feature_matrix(df, feature_columns=["feat_a", "feat_b"])
    assert "Missing required feature columns" in str(exc.value)

    schema = FeatureSchema(
        feature_names=["feat_a", "feat_b"],
        numeric_features=["feat_a", "feat_b"],
    )
    with pytest.raises(ValueError) as exc:
        schema.validate(df)
    assert "missing required features" in str(exc.value)


def test_non_numeric_feature_handling():
    """Test 5: Explicit handling of non-numeric columns without silent coercion."""
    df = pd.DataFrame(
        {
            "num_feat": [1.0, 2.0],
            "str_feat": ["flag_A", "flag_B"],
            "Label": [0, 1],
        }
    )

    X_dropped, _, schema = prepare_feature_matrix(df, drop_non_numeric=True)
    assert "str_feat" not in X_dropped.columns
    assert list(X_dropped.columns) == ["num_feat"]

    with pytest.raises(TypeError) as exc:
        prepare_feature_matrix(df, drop_non_numeric=False)
    assert "Non-numeric columns detected" in str(exc.value)


def test_metadata_exclusion():
    """Verify that identifiers and timestamps are excluded to prevent target leakage."""
    df = pd.DataFrame(
        {
            "Flow ID": ["192.168.1.1-10.0.0.1", "192.168.1.2-10.0.0.1"],
            "Source IP": ["192.168.1.1", "192.168.1.2"],
            "Timestamp": ["07/07/2017 01:00", "07/07/2017 01:05"],
            "packet_count": [10, 50],
            "Label": [0, 1],
        }
    )
    X, _, schema = prepare_feature_matrix(df, drop_metadata=True)
    assert "Flow ID" not in X.columns
    assert "Source IP" not in X.columns
    assert "Timestamp" not in X.columns
    assert list(X.columns) == ["packet_count"]
    assert "Flow ID" in schema.metadata_excluded


def test_duplicate_column_rejection():
    """Test 7: Duplicate column names are strictly rejected."""
    df = pd.DataFrame([[1, 2, 0]], columns=["feat_a", "feat_a", "Label"])
    with pytest.raises(ValueError) as exc:
        prepare_feature_matrix(df)
    assert "Duplicate column names detected" in str(exc.value)


def test_all_missing_column_rejection():
    """Test Phase 2.3: Completely missing (100% NaN) columns are rejected."""
    df = pd.DataFrame(
        {
            "feat_good": [1.0, 2.0],
            "feat_empty": [np.nan, np.nan],
            "Label": [0, 1],
        }
    )
    with pytest.raises(ValueError) as exc:
        prepare_feature_matrix(df)
    assert "100% missing values" in str(exc.value)


def test_schema_extra_columns_policy():
    """Test FeatureSchema handling of extra columns according to allow_extra_columns policy."""
    schema = FeatureSchema(
        feature_names=["f1", "f2"],
        numeric_features=["f1", "f2"],
    )
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "extra_col": [9, 9]})

    # When allow_extra_columns=True (default): extra column is cleanly dropped
    aligned = schema.validate(df, allow_extra_columns=True)
    assert list(aligned.columns) == ["f1", "f2"]

    # When allow_extra_columns=False: ValueError is raised
    with pytest.raises(ValueError) as exc:
        schema.validate(df, allow_extra_columns=False)
    assert "Unexpected additional columns" in str(exc.value)
