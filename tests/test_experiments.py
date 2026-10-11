"""
Automated Unit Tests for src.experiments (Milestone 16).
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.experiments import (
    compute_per_family_recall,
    build_candidate_estimators,
    evaluate_model_comprehensive,
)


def test_compute_per_family_recall_basic():
    """Verify compute_per_family_recall accurately computes per-label metrics."""
    y_true_raw = ["BENIGN", "BENIGN", "DDoS", "DDoS", "FTP-Patator"]
    y_pred_binary = [0, 1, 1, 1, 0]  # 1 FP in benign, 2 TP in DDoS, 1 FN in FTP-Patator

    results = compute_per_family_recall(y_true_raw, y_pred_binary, pos_label=1)

    assert "BENIGN" in results
    assert results["BENIGN"]["category"] == "benign"
    assert results["BENIGN"]["total_samples"] == 2
    assert results["BENIGN"]["predicted_benign"] == 1
    assert results["BENIGN"]["predicted_attack"] == 1
    assert results["BENIGN"]["false_positive_rate"] == 0.5
    assert results["BENIGN"]["benign_specificity"] == 0.5

    assert "DDoS" in results
    assert results["DDoS"]["category"] == "attack"
    assert results["DDoS"]["total_samples"] == 2
    assert results["DDoS"]["detected_attacks"] == 2
    assert results["DDoS"]["missed_attacks"] == 0
    assert results["DDoS"]["recall"] == 1.0

    assert "FTP-Patator" in results
    assert results["FTP-Patator"]["category"] == "attack"
    assert results["FTP-Patator"]["total_samples"] == 1
    assert results["FTP-Patator"]["detected_attacks"] == 0
    assert results["FTP-Patator"]["missed_attacks"] == 1
    assert results["FTP-Patator"]["recall"] == 0.0


def test_compute_per_family_recall_length_mismatch():
    """Verify compute_per_family_recall raises ValueError on length mismatch."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_per_family_recall(["BENIGN"], [0, 1])


def test_build_candidate_estimators():
    """Verify build_candidate_estimators instantiates expected model dictionary."""
    estimators = build_candidate_estimators(random_state=42, n_estimators=50, max_depth=10)

    expected_keys = {
        "rf_unweighted",
        "rf_balanced",
        "rf_balanced_subsample",
        "hist_gradient_boosting",
        "extra_trees_balanced",
    }
    assert set(estimators.keys()) == expected_keys

    # Check class weights
    assert estimators["rf_unweighted"].class_weight is None
    assert estimators["rf_balanced"].class_weight == "balanced"
    assert estimators["rf_balanced_subsample"].class_weight == "balanced_subsample"
    assert estimators["hist_gradient_boosting"].class_weight == "balanced"
    assert estimators["extra_trees_balanced"].class_weight == "balanced"


def test_evaluate_model_comprehensive_synthetic():
    """Verify evaluate_model_comprehensive computes expected metric keys."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 4), columns=[f"f_{i}" for i in range(4)])
    y_raw = pd.Series(["BENIGN"] * 10 + ["DDoS"] * 10)
    y_binary = pd.Series([0] * 10 + [1] * 10)

    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    pipeline.fit(X, y_binary)

    metrics = evaluate_model_comprehensive(
        pipeline=pipeline,
        X=X,
        y_binary=y_binary,
        y_raw=y_raw,
        dataset_name="Synthetic Test",
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "per_family_recall" in metrics
    assert "BENIGN" in metrics["per_family_recall"]
    assert "DDoS" in metrics["per_family_recall"]
    assert metrics["dataset_name"] == "Synthetic Test"
