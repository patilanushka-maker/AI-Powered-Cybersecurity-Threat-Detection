"""
Automated unit tests for src/evaluate.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_binary_metrics,
    evaluate_model,
    save_evaluation_report,
)


def test_metric_correctness_known_synthetic_values():
    """
    Test 13: Exact mathematical verification of security evaluation metrics.
    Ground truth: 4 benign (0), 4 attack (1).
    Predictions:
    - True Negatives (0 -> 0): 3
    - False Positives (0 -> 1): 1
    - False Negatives (1 -> 0): 1
    - True Positives (1 -> 1): 3
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.85, 0.7, 0.4])

    metrics = compute_binary_metrics(y_true, y_pred, y_prob=y_prob, pos_label=1)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == 0.75
    # False Positive Rate = FP / (FP + TN) = 1 / (1 + 3) = 0.25
    assert metrics["false_positive_rate"] == 0.25

    # Check confusion matrix counts
    cm_info = metrics["confusion_matrix"]
    assert cm_info["true_negatives"] == 3
    assert cm_info["false_positives"] == 1
    assert cm_info["false_negatives"] == 1
    assert cm_info["true_positives"] == 3
    assert cm_info["matrix"] == [[3, 1], [1, 3]]


def test_confusion_matrix_shape_and_ordering():
    """Test 14 & 15: Matrix shape is 2x2 with order [[TN, FP], [FN, TP]]."""
    y_true = [0, 1]
    y_pred = [0, 1]
    metrics = compute_binary_metrics(y_true, y_pred, pos_label=1)

    matrix = metrics["confusion_matrix"]["matrix"]
    assert len(matrix) == 2
    assert len(matrix[0]) == 2
    assert matrix == [[1, 0], [0, 1]]


def test_evaluate_model_with_pipeline_and_plot(tmp_path: Path):
    """Test evaluate_model with fitted estimator and plot generation."""
    class DummyModel:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    X_test = pd.DataFrame({"f": [1, 2]})
    y_test = pd.Series([0, 1])
    plot_file = tmp_path / "cm_plot.png"

    results = evaluate_model(
        DummyModel(),
        X_test,
        y_test,
        pos_label=1,
        save_confusion_matrix_plot=plot_file,
    )

    assert results["accuracy"] == 1.0
    assert results["false_positive_rate"] == 0.0
    assert plot_file.exists()


def test_save_evaluation_report(tmp_path: Path):
    """Test saving metrics dictionary to JSON report."""
    metrics = {"accuracy": 0.95, "f1_score": 0.94}
    report_path = tmp_path / "eval_report.json"

    saved = save_evaluation_report(metrics, report_path, overwrite=False)
    assert saved.exists()

    with pytest.raises(FileExistsError):
        save_evaluation_report(metrics, report_path, overwrite=False)


def test_compute_binary_metrics_non_binary_raises():
    """Verify ValueError when y_true contains values outside {0, 1}."""
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]
    with pytest.raises(ValueError) as exc:
        compute_binary_metrics(y_true, y_pred)
    assert "non-binary labels" in str(exc.value)


def test_compute_binary_metrics_single_class_ground_truth():
    """Verify that when only one class exists in ground truth, ROC-AUC is safely None."""
    # Only benign (0) in ground truth
    y_true_benign = [0, 0, 0, 0]
    y_pred = [0, 0, 1, 0]
    y_prob = [0.1, 0.2, 0.8, 0.3]
    metrics_benign = compute_binary_metrics(y_true_benign, y_pred, y_prob=y_prob)
    assert metrics_benign["roc_auc"] is None
    assert metrics_benign["pr_auc"] is None
    assert metrics_benign["roc_auc_applicable"] is False
    assert metrics_benign["accuracy"] == 0.75

    # Only attack (1) in ground truth
    y_true_attack = [1, 1, 1]
    y_pred_attack = [1, 1, 0]
    metrics_attack = compute_binary_metrics(y_true_attack, y_pred_attack, y_prob=[0.9, 0.8, 0.3])
    assert metrics_attack["roc_auc"] is None
    assert metrics_attack["roc_auc_applicable"] is False
    assert metrics_attack["recall"] == pytest.approx(2 / 3)

