"""
Tests for Milestone 19: Capture-Level Generalization and Robustness Module.

Validates:
- Capture provenance preservation and exclusion from feature matrices.
- Train/test capture separation and cross-split leakage detection.
- Feature-vector duplicate detection.
- Conflicting labels on identical feature vectors detection.
- Metric calculation, confusion matrix alignment, and per-family count reconciliation.
"""

import numpy as np
import pandas as pd
import pytest

from src.capture_generalization import (
    audit_capture_provenance,
    detect_feature_duplicates,
    detect_cross_split_leakage,
    compute_comprehensive_metrics,
)
from src.feature_engineering import prepare_feature_matrix


def test_capture_provenance_preservation_and_metadata_exclusion(tmp_path):
    """
    Test that capture provenance column is properly tracked on loading,
    and excluded from model feature matrices by prepare_feature_matrix.
    """
    # Create two synthetic capture CSVs
    cap1_path = tmp_path / "CaptureA.csv"
    cap2_path = tmp_path / "CaptureB.csv"

    df1 = pd.DataFrame({
        "Flow Duration": [100, 200, 300],
        "Total Fwd Packets": [1, 2, 3],
        "Label": ["BENIGN", "BENIGN", "DDoS"],
    })
    df2 = pd.DataFrame({
        "Flow Duration": [400, 500],
        "Total Fwd Packets": [4, 5],
        "Label": ["FTP-Patator", "BENIGN"],
    })

    df1.to_csv(cap1_path, index=False)
    df2.to_csv(cap2_path, index=False)

    combined_df, audit_info = audit_capture_provenance(
        [cap1_path, cap2_path], provenance_col="capture_source"
    )

    assert audit_info["provenance_column_present_in_combined"] is True
    assert audit_info["total_combined_rows"] == 5
    assert audit_info["per_file_original_rows"]["CaptureA.csv"] == 3
    assert audit_info["per_file_original_rows"]["CaptureB.csv"] == 2
    assert "capture_source" in combined_df.columns

    # Verify prepare_feature_matrix drops capture_source from X
    X, y, schema = prepare_feature_matrix(combined_df, target_column="Label", drop_metadata=True)
    assert "capture_source" not in X.columns
    assert "capture_source" in schema.metadata_excluded
    assert "Label" not in X.columns
    assert len(X) == 5
    assert len(y) == 5


def test_duplicate_detection_and_ratio():
    """
    Test accurate quantification of duplicate feature vectors.
    """
    # 4 rows, 2 identical vectors (rows 0 and 1)
    df_feats = pd.DataFrame({
        "Feature1": [10.0, 10.0, 20.0, 30.0],
        "Feature2": [1.0, 1.0, 2.0, 3.0],
    })
    labels = pd.Series(["BENIGN", "BENIGN", "BENIGN", "DDoS"])

    audit = detect_feature_duplicates(df_feats, labels)

    assert audit["total_rows"] == 4
    assert audit["duplicate_rows_count"] == 2
    assert audit["duplicate_rows_ratio"] == 0.5
    assert audit["unique_duplicate_feature_vectors"] == 1
    assert audit["conflicting_labels_count"] == 0
    assert audit["conflicting_labels_detected"] is False


def test_conflicting_labels_on_identical_feature_vectors():
    """
    Test detection of conflicting labels when identical feature vectors have different classes.
    """
    # 4 rows, rows 0 and 1 have identical features but conflicting labels
    df_feats = pd.DataFrame({
        "Feature1": [10.0, 10.0, 20.0, 30.0],
        "Feature2": [1.0, 1.0, 2.0, 3.0],
    })
    # Conflicting labels for [10.0, 1.0]
    labels_conflict = pd.Series(["BENIGN", "DDoS", "BENIGN", "DDoS"])

    audit_conflict = detect_feature_duplicates(df_feats, labels_conflict)
    assert audit_conflict["conflicting_labels_count"] == 1
    assert audit_conflict["conflicting_labels_detected"] is True

    # Non-conflicting case
    labels_clean = pd.Series(["BENIGN", "BENIGN", "BENIGN", "DDoS"])
    audit_clean = detect_feature_duplicates(df_feats, labels_clean)
    assert audit_clean["conflicting_labels_count"] == 0
    assert audit_clean["conflicting_labels_detected"] is False


def test_cross_split_leakage_detection():
    """
    Test detection of feature vectors crossing train/validation boundaries.
    """
    X_train = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0, 4.0],
        "Feature2": [10.0, 20.0, 30.0, 40.0],
    })
    # Validation contains one vector identical to row 0 in train ([1.0, 10.0])
    # and two novel vectors
    X_val = pd.DataFrame({
        "Feature1": [1.0, 5.0, 6.0],
        "Feature2": [10.0, 50.0, 60.0],
    })

    leakage = detect_cross_split_leakage(X_train, X_val)
    assert leakage["val_total_rows"] == 3
    assert leakage["leaked_val_rows_count"] == 1
    assert pytest.approx(leakage["leaked_val_rows_ratio"], 0.001) == 1 / 3
    assert leakage["unique_leaked_vectors_count"] == 1
    assert leakage["leakage_detected"] is True
    assert list(leakage["val_leaked_mask"]) == [True, False, False]


def test_metric_and_class_count_reconciliation():
    """
    Test that comprehensive metrics calculate accurate confusion matrix terms,
    and that per-family counts reconcile strictly with total dataset counts.
    """
    # 10 samples: 6 Benign (0), 4 Attack (1)
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    # Predictions: 1 false positive, 1 false negative
    y_pred = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.8, 0.4, 0.9, 0.85, 0.95])
    raw_labels = pd.Series([
        "BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN", "BENIGN",
        "DDoS", "DDoS", "FTP-Patator", "SSH-Patator"
    ])
    trained_families = {"DDoS", "FTP-Patator"}

    metrics = compute_comprehensive_metrics(
        y_true, y_pred, y_prob, raw_labels, trained_families=trained_families
    )

    # Reconciliation checks
    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 5
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 3
    assert cm["true_negatives"] + cm["false_positives"] == 6  # total benign
    assert cm["true_positives"] + cm["false_negatives"] == 4  # total attack
    assert metrics["total_samples"] == 10

    assert pytest.approx(metrics["accuracy"], 0.001) == 8 / 10
    assert pytest.approx(metrics["precision"], 0.001) == 3 / 4
    assert pytest.approx(metrics["recall"], 0.001) == 3 / 4
    assert pytest.approx(metrics["false_positive_rate"], 0.001) == 1 / 6
    assert metrics["roc_auc"] is not None
    assert metrics["average_precision_pr_auc"] is not None

    # Check per-family breakdown
    fam = metrics["per_family_breakdown"]
    assert fam["BENIGN"]["total_samples"] == 6
    assert fam["BENIGN"]["correctly_identified_benign"] == 5
    assert fam["BENIGN"]["false_positives"] == 1

    assert fam["DDOS"]["total_samples"] == 2
    assert fam["DDOS"]["detected_attacks"] == 1
    assert fam["DDOS"]["missed_attacks"] == 1
    assert fam["DDOS"]["present_in_training"] is True

    assert fam["FTP-PATATOR"]["total_samples"] == 1
    assert fam["FTP-PATATOR"]["detected_attacks"] == 1
    assert fam["FTP-PATATOR"]["present_in_training"] is True

    assert fam["SSH-PATATOR"]["total_samples"] == 1
    assert fam["SSH-PATATOR"]["detected_attacks"] == 1
    assert fam["SSH-PATATOR"]["present_in_training"] is False  # not in trained_families


def test_train_test_capture_separation_disjoint():
    """
    Test that when training and evaluation splits are disjoint captures,
    cross-split leakage detection reports zero leakage.
    """
    X_train_friday = pd.DataFrame({
        "Flow Duration": [10.0, 20.0, 30.0],
        "Total Fwd Packets": [1.0, 2.0, 3.0],
    })
    X_test_tuesday = pd.DataFrame({
        "Flow Duration": [100.0, 200.0, 300.0],
        "Total Fwd Packets": [10.0, 20.0, 30.0],
    })

    leakage = detect_cross_split_leakage(X_train_friday, X_test_tuesday)
    assert leakage["leaked_val_rows_count"] == 0
    assert leakage["leaked_val_rows_ratio"] == 0.0
    assert leakage["unique_leaked_vectors_count"] == 0
    assert leakage["leakage_detected"] is False
