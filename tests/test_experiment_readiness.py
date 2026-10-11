"""
Automated unit and integration tests for experiment readiness and baseline pipeline guarantees.

Covers:
- Strict target label policy (unknown, missing, blank, and CIC-IDS2017 label handling)
- Feature schema persistence and training/inference consistency
- Target and metadata leakage prevention
- Rare-class and single-class split and metric behavior
- Reproducible, stratified controlled dataset sampling
- Empty and invalid training data rejection
- Verification that dataset validation has zero training side effects
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    encode_binary_target,
    sample_dataset,
    train_test_split_network_flows,
    split_data_temporal,
    DEFAULT_BENIGN_LABELS,
    DEFAULT_RECOGNIZED_ATTACK_LABELS,
)
from src.feature_engineering import (
    prepare_feature_matrix,
    FeatureSchema,
    DEFAULT_METADATA_COLUMNS,
)
from src.train import (
    build_training_pipeline,
    train_model,
    save_model_artifact,
    load_model_artifact,
)
from src.evaluate import compute_binary_metrics, evaluate_model
from src.dataset_validator import validate_dataset_directory
from main import main


def test_unknown_labels_rejected_with_actionable_error():
    """Verify that unknown attack or benign-like labels are rejected, never silently mapped."""
    labels = pd.Series(["BENIGN", "BENIGN", "UNKNOWN_TROJAN_XZY"])
    with pytest.raises(ValueError) as exc:
        encode_binary_target(labels, reject_unknown=True)
    assert "Unknown or unexpected label(s) detected" in str(exc.value)
    assert "UNKNOWN_TROJAN_XZY" in str(exc.value)


def test_missing_and_blank_labels_rejected():
    """Verify that missing (NaN) and blank string labels are rejected, never silently mapped."""
    # Null / NaN labels
    labels_null = pd.Series(["BENIGN", None, "DDoS"])
    with pytest.raises(ValueError) as exc1:
        encode_binary_target(labels_null)
    assert "null or missing values" in str(exc1.value).lower()

    # Empty string labels
    labels_empty = pd.Series(["BENIGN", "", "DDoS"])
    with pytest.raises(ValueError) as exc2:
        encode_binary_target(labels_empty)
    assert "empty string labels" in str(exc2.value).lower()

    # Whitespace-only string labels
    labels_whitespace = pd.Series(["BENIGN", "    ", "DDoS"])
    with pytest.raises(ValueError) as exc3:
        encode_binary_target(labels_whitespace)
    assert "empty string labels" in str(exc3.value).lower()


def test_cic_ids_specific_attack_labels_recognized():
    """Verify recognition of canonical CIC-IDS2017 attack labels including dash variations."""
    test_labels = pd.Series(
        [
            "BENIGN",
            "Web Attack – Brute Force",  # Unicode en-dash
            "Web Attack - XSS",          # ASCII hyphen
            "Web Attack - Sql Injection",
            "Infiltration",
            "Bot",
            "DDoS",
            "PortScan",
            "FTP-Patator",
            "SSH-Patator",
            "DoS Slowloris",
            "DoS Slowhttptest",
            "DoS Hulk",
            "DoS GoldenEye",
            "Heartbleed",
        ]
    )
    encoded = encode_binary_target(test_labels)
    # BENIGN is 0, all 14 attacks are 1
    assert encoded[0] == 0
    assert (encoded[1:] == 1).all()
    assert len(encoded) == 15


def test_target_column_leakage_strictly_prevented():
    """Verify that the target column cannot leak into features in any pipeline stage."""
    df = pd.DataFrame(
        {
            "Duration": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "Packets": [10, 20, 30, 40, 50, 60],
            "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS", "BENIGN", "DDoS"],
        }
    )

    # 1. prepare_feature_matrix
    X, y, schema = prepare_feature_matrix(df, target_column="Label")
    assert "Label" not in X.columns
    assert "Label" not in schema.feature_names
    assert schema.target_column == "Label"

    # 2. FeatureSchema validation
    aligned = schema.validate(df)  # df includes 'Label'
    assert "Label" not in aligned.columns

    # 3. train_test_split_network_flows
    X_train, X_test, y_train, y_test = train_test_split_network_flows(
        X, y, test_size=0.33, stratify=False
    )
    assert "Label" not in X_train.columns
    assert "Label" not in X_test.columns


def test_metadata_columns_strictly_excluded():
    """Verify that IP addresses, timestamps, and flow IDs are excluded from features."""
    df = pd.DataFrame(
        {
            "Flow ID": ["id1", "id2"],
            "Source IP": ["192.168.1.1", "192.168.1.2"],
            "Destination IP": ["10.0.0.1", "10.0.0.2"],
            "Timestamp": ["07/07/2017 10:00", "07/07/2017 10:01"],
            "Flow Duration": [100.0, 200.0],
            "Label": ["BENIGN", "DDoS"],
        }
    )
    X, y, schema = prepare_feature_matrix(df, target_column="Label", drop_metadata=True)
    assert list(X.columns) == ["Flow Duration"]
    for meta in ["Flow ID", "Source IP", "Destination IP", "Timestamp"]:
        assert meta not in X.columns
        assert meta in schema.metadata_excluded


def test_feature_schema_persistence_and_deserialization(tmp_path: Path):
    """Verify that FeatureSchema persists inside model artifact and reconstructs perfectly."""
    schema = FeatureSchema(
        feature_names=["f1", "f2", "f3"],
        numeric_features=["f1", "f2", "f3"],
        metadata_excluded=["Flow ID", "Source IP"],
        target_column="Label",
    )
    X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]})
    y_train = pd.Series([0, 1])

    pipeline, metadata = train_model(X_train, y_train, feature_schema=schema)
    assert metadata["feature_schema"] is not None

    artifact_file = tmp_path / "model.joblib"
    save_model_artifact(pipeline, metadata, artifact_file)

    loaded_pipe, loaded_meta = load_model_artifact(artifact_file)
    reconstructed_schema = FeatureSchema.from_dict(loaded_meta["feature_schema"])
    assert reconstructed_schema.feature_names == ["f1", "f2", "f3"]
    assert reconstructed_schema.metadata_excluded == ["Flow ID", "Source IP"]
    assert reconstructed_schema.target_column == "Label"


def test_consistent_imputation_learned_from_train_only():
    """Verify that numerical median imputer learns medians exclusively on training data."""
    X_train = pd.DataFrame({"feat": [10.0, 20.0, 30.0]})  # Median = 20.0
    y_train = pd.Series([0, 0, 1])

    pipeline, _ = train_model(X_train, y_train)

    # Impute test data containing NaN
    X_test = pd.DataFrame({"feat": [np.nan, 100.0]})
    transformed_test = pipeline.named_steps["imputer"].transform(X_test)
    assert transformed_test[0, 0] == 20.0  # Median learned from train, not test!


def test_rare_class_stratified_split_failure_behavior():
    """Verify that stratified splitting raises ValueError when minority class has < 2 samples."""
    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y_rare = pd.Series([0, 0, 0, 1])  # Class 1 has only 1 sample
    with pytest.raises(ValueError) as exc:
        train_test_split_network_flows(X, y_rare, stratify=True)
    assert "minority class has fewer than 2 samples" in str(exc.value)


def test_missing_class_metric_behavior():
    """Verify evaluation metric behavior when a class is absent in held-out ground truth."""
    y_true = np.array([0, 0, 0, 0])  # Only Benign present
    y_pred = np.array([0, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.3])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob=y_prob, pos_label=1)
    assert metrics["sample_count"] == 4
    assert metrics["roc_auc_applicable"] is False
    assert metrics["roc_auc"] is None  # Cannot compute ROC-AUC with single class
    assert metrics["false_positive_rate"] == 0.25  # 1 FP / 4 TN = 0.25
    assert metrics["recall"] == 0.0  # 0 TP / 0 Positives -> zero_division=0.0


def test_controlled_dataset_sampling_reproducibility():
    """Verify that sample_dataset is reproducible, stratified, and records full provenance."""
    df = pd.DataFrame(
        {
            "feat": range(100),
            "Label": ["BENIGN"] * 80 + ["DDoS"] * 20,
        }
    )

    # Sample to 30 rows
    sampled_1, meta_1 = sample_dataset(df, max_samples=30, target_column="Label", random_state=42)
    sampled_2, meta_2 = sample_dataset(df, max_samples=30, target_column="Label", random_state=42)

    pd.testing.assert_frame_equal(sampled_1, sampled_2)
    assert len(sampled_1) == 30
    assert meta_1["sampling_applied"] is True
    assert meta_1["total_rows_available"] == 100
    assert meta_1["rows_retained"] == 30
    assert meta_1["sampling_method"] == "stratified"
    # Class ratio preserved: 80% Benign (24 rows), 20% DDoS (6 rows)
    assert (sampled_1["Label"] == "BENIGN").sum() == 24
    assert (sampled_1["Label"] == "DDoS").sum() == 6

    # Sample when max_samples >= len(df) -> no sampling applied, no rows dropped
    sampled_full, meta_full = sample_dataset(df, max_samples=150, target_column="Label")
    assert meta_full["sampling_applied"] is False
    assert len(sampled_full) == 100

    # Invalid max_samples
    with pytest.raises(ValueError, match="positive integer"):
        sample_dataset(df, max_samples=0)


def test_empty_or_single_class_training_data_rejected():
    """Verify that empty datasets and single-class datasets are rejected prior to fitting."""
    empty_df = pd.DataFrame()
    empty_series = pd.Series(dtype=int)
    with pytest.raises(ValueError, match="Cannot train model on empty"):
        train_model(empty_df, empty_series)

    X = pd.DataFrame({"f": [1, 2, 3]})
    y_single = pd.Series([0, 0, 0])
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        train_model(X, y_single)


def test_validation_has_zero_training_side_effects(tmp_path: Path):
    """Verify that running dataset validation produces no trained models or .joblib files."""
    data_dir = tmp_path / "val_data"
    data_dir.mkdir()
    df = pd.DataFrame({"f1": [1.0, 2.0], "Label": ["BENIGN", "DDoS"]})
    df.to_csv(data_dir / "flows.csv", index=False)

    report_path = tmp_path / "report.json"
    rep = validate_dataset_directory(data_dir, report_out=report_path)

    assert rep["model_training_performed"] is False
    assert rep["model_evaluation_performed"] is False
    assert list(tmp_path.glob("*.joblib")) == []


def test_sample_dataset_preserves_rare_classes_against_rounding():
    """Verify sample_dataset guarantees rare class preservation when rounding would drop it."""
    # 10,000 Benign and 2 DDoS rows (DDoS is 0.02% of data)
    df = pd.DataFrame(
        {
            "feat": range(10002),
            "Label": ["BENIGN"] * 10000 + ["DDoS"] * 2,
        }
    )

    # Standard train_test_split(train_size=1000) rounds 2 * (1000/10002) = 0.199 down to 0
    # Our enhanced sample_dataset must guarantee both DDoS rows are preserved
    sampled, meta = sample_dataset(df, max_samples=1000, target_column="Label", random_state=42)

    assert len(sampled) == 1000
    assert meta["sampling_applied"] is True
    assert meta["classes_preserved"] is True
    assert (sampled["Label"] == "DDoS").sum() == 2
    assert (sampled["Label"] == "BENIGN").sum() == 998
    assert meta["class_distribution_before"] == {"BENIGN": 10000, "DDoS": 2}
    assert meta["class_distribution_after"] == {"BENIGN": 998, "DDoS": 2}


def test_sample_dataset_max_samples_smaller_than_unique_classes():
    """Verify sample_dataset raises ValueError when max_samples < number of distinct classes."""
    df = pd.DataFrame(
        {
            "feat": [1, 2, 3, 4],
            "Label": ["BENIGN", "DDoS", "PortScan", "Bot"],
        }
    )
    with pytest.raises(ValueError, match="cannot be smaller than the number of unique classes"):
        sample_dataset(df, max_samples=2, target_column="Label")
