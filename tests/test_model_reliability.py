"""
Unit Tests for Milestone 20: Model Reliability and Generalization-Focused Evaluation.

Validates:
- Positive-class probability resolution and dynamic class ordering.
- Group-aware split separation (strictly zero overlapping groups).
- Exact feature-vector duplicate detection and conflicting-label reporting.
- Feature-column alignment and schema verification.
- Threshold consistency and monotonicity.
- Confusion matrix reconciliation.
- Attack family coverage and explicit unseen class warnings.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.model_reliability import (
    resolve_positive_class_index,
    get_positive_class_probabilities,
    exact_duplicate_detection,
    perform_group_aware_split,
    evaluate_threshold_spectrum,
    compute_attack_family_coverage,
)


def test_positive_class_probability_resolution():
    """
    Test that class ordering is dynamically resolved from estimator classes_
    rather than assumed to be index 1.
    """
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])

    # Case A: Standard DummyClassifier
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X, y)
    idx = resolve_positive_class_index(clf, pos_label=1)
    assert idx == list(clf.classes_).index(1)

    # Case B: In a Pipeline
    pipe = Pipeline([("imputer", SimpleImputer()), ("classifier", clf)])
    pipe.fit(X, y)
    idx_pipe = resolve_positive_class_index(pipe, pos_label=1)
    assert idx_pipe == list(clf.classes_).index(1)

    # Case C: Missing pos_label raises ValueError
    with pytest.raises(ValueError, match="not found in estimator classes"):
        resolve_positive_class_index(clf, pos_label=999)


def test_get_positive_class_probabilities_bounds():
    """
    Test that extracted probabilities are finite and strictly within [0.0, 1.0].
    """
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 0, 1])
    clf = DummyClassifier(strategy="prior")
    clf.fit(X, y)

    probs = get_positive_class_probabilities(clf, X, pos_label=1)
    assert len(probs) == 4
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_exact_duplicate_detection_and_conflicts():
    """
    Test exact duplicate detection and verify conflicting labels are reported
    without dropping rows.
    """
    # 4 rows, rows 0 and 1 are identical with identical labels
    # rows 2 and 3 are identical with CONFLICTING labels
    X = pd.DataFrame({
        "colA": [10.0, 10.0, 20.0, 20.0],
        "colB": [1.0, 1.0, 2.0, 2.0],
    })
    y = pd.Series(["BENIGN", "BENIGN", "BENIGN", "DDoS"])

    audit = exact_duplicate_detection(X, y)

    assert audit["total_rows"] == 4
    assert audit["duplicate_rows_count"] == 4
    assert audit["unique_duplicate_vectors_count"] == 2
    assert audit["conflicting_label_groups_count"] == 1
    assert audit["conflicting_labels_detected"] is True
    assert len(audit["conflicting_groups_details"]) == 1


def test_group_aware_split_separation_and_zero_leakage():
    """
    Test that group-aware splitting guarantees zero identical feature vectors
    cross the train/validation boundary.
    """
    # 6 rows: 3 pairs of duplicate feature vectors
    X = pd.DataFrame({
        "colA": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "colB": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
    })
    y_bin = pd.Series([0, 0, 1, 1, 0, 0])
    y_raw = pd.Series(["BENIGN", "BENIGN", "DDoS", "DDoS", "BENIGN", "BENIGN"])

    split_res = perform_group_aware_split(X, y_bin, y_raw, test_size=0.33, random_state=42)

    assert split_res["total_unique_feature_groups"] == 3
    assert split_res["overlapping_groups_count"] == 0
    assert split_res["group_separation_verified"] is True
    assert split_res["conflicting_binary_groups_count"] == 0

    # Verify that no row in train has identical features to any row in val
    train_idx = split_res["train_indices"]
    val_idx = split_res["val_indices"]
    train_tuples = set(tuple(x) for x in X.iloc[train_idx].itertuples(index=False, name=None))
    val_tuples = set(tuple(x) for x in X.iloc[val_idx].itertuples(index=False, name=None))
    assert train_tuples.isdisjoint(val_tuples)


def test_threshold_spectrum_monotonicity_and_reconciliation():
    """
    Test threshold spectrum evaluation:
    - Predicted alerts must be monotonically non-increasing as threshold increases.
    - Confusion matrix entries must reconcile to total row count.
    """
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 0, 1, 1, 1])

    # Simple model with predictable probabilities
    class MockEstimator:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            # Returns fixed scores: 0.1, 0.35, 0.45, 0.65, 0.85
            scores = np.array([0.1, 0.35, 0.45, 0.65, 0.85])
            return np.column_stack([1 - scores, scores])

    mock_model = MockEstimator()
    thresholds = [0.20, 0.40, 0.60, 0.80]
    res = evaluate_threshold_spectrum(mock_model, X, y, thresholds=thresholds, pos_label=1)

    alert_counts = [res[f"threshold_{t:.2f}"]["predicted_alerts_count"] for t in thresholds]
    # Check monotonicity
    assert alert_counts == sorted(alert_counts, reverse=True)
    # Exact alert counts:
    # >= 0.20: 4 alerts (0.35, 0.45, 0.65, 0.85)
    # >= 0.40: 3 alerts (0.45, 0.65, 0.85)
    # >= 0.60: 2 alerts (0.65, 0.85)
    # >= 0.80: 1 alert (0.85)
    assert alert_counts == [4, 3, 2, 1]

    # Reconciliation for each threshold
    for t in thresholds:
        cm = res[f"threshold_{t:.2f}"]["confusion_matrix"]
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == len(y)


def test_attack_family_coverage_and_unseen_warnings():
    """
    Test attack family coverage breakdown and ensure explicit warnings are attached
    to classes absent from training.
    """
    y_true_bin = np.array([0, 0, 1, 1, 1])
    y_pred_bin = np.array([0, 0, 1, 0, 0])
    y_raw = pd.Series(["BENIGN", "BENIGN", "DDoS", "DoS GoldenEye", "Heartbleed"])
    trained_families = {"DDoS"}

    coverage = compute_attack_family_coverage(
        y_true_bin, y_pred_bin, y_raw, trained_families=trained_families
    )

    # DDoS is in training
    assert coverage["DDOS"]["represented_in_training"] is True
    assert coverage["DDOS"]["warning"] is None
    assert coverage["DDOS"]["true_positives"] == 1

    # GoldenEye is NOT in training
    assert coverage["DOS GOLDENEYE"]["represented_in_training"] is False
    assert coverage["DOS GOLDENEYE"]["warning"] is not None
    assert "EXPLICIT WARNING" in coverage["DOS GOLDENEYE"]["warning"]
    assert coverage["DOS GOLDENEYE"]["recall"] == 0.0

    # Heartbleed is NOT in training
    assert coverage["HEARTBLEED"]["represented_in_training"] is False
    assert coverage["HEARTBLEED"]["warning"] is not None
    assert "EXPLICIT WARNING" in coverage["HEARTBLEED"]["warning"]


def test_saved_model_checksum_preservation():
    """
    Test that evaluated model artifacts maintain immutable SHA-256 checksums,
    confirming no models were modified or overwritten during audits.
    """
    from src.model_reliability import compute_file_sha256

    expected_checksums = {
        "models/random_forest_multi_attack.joblib": "486d6c75f00a163204cb993d0a4e9e2688c1b22ef58cd58e2e53e418d7fcd35f",
        "models/model_extra_trees_balanced_m16.joblib": "c60cef05d9ad7751f013e13d5c297dc2e9b1191ec9aaf1af887cb16227c7f60b",
        "models/random_forest_baseline.joblib": "e6e6d785ab068381befcfbc914402be9253cec6789ac55c820952b670a5bc3b3",
    }

    for path_str, expected_hash in expected_checksums.items():
        actual_hash = compute_file_sha256(path_str)
        assert actual_hash == expected_hash, f"Model artifact {path_str} checksum changed!"


def test_confusion_matrix_arithmetic():
    """
    Test confusion matrix arithmetic invariance:
    TN + FP + FN + TP == total samples,
    TPR = TP / (TP + FN),
    FPR = FP / (FP + TN).
    """
    tn, fp, fn, tp = 15921, 0, 83, 4117
    total = tn + fp + fn + tp
    assert total == 20121

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    assert pytest.approx(tpr, 0.0001) == 0.980238
    assert fpr == 0.0

