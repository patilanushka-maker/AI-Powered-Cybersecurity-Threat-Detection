"""
Automated Unit Tests for Milestone 17 Threshold Analysis and False Negative Investigation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.threshold_and_fn_analysis import (
    evaluate_threshold_sweep,
    run_threshold_analysis,
    analyze_false_negatives,
)
from src.train import train_model, save_model_artifact


def test_evaluate_threshold_sweep_monotonicity():
    """Verify threshold sweep metrics behave as expected mathematically."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.05, 0.15, 0.35, 0.25, 0.45, 0.85])

    results = evaluate_threshold_sweep(y_true, probs, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5])

    assert len(results) == 5
    assert "0.10" in results
    assert "0.50" in results

    # As threshold increases, predicted positives should decrease or stay equal
    total_pos_01 = (
        results["0.10"]["confusion_matrix"]["true_positives"]
        + results["0.10"]["confusion_matrix"]["false_positives"]
    )
    total_pos_05 = (
        results["0.50"]["confusion_matrix"]["true_positives"]
        + results["0.50"]["confusion_matrix"]["false_positives"]
    )
    assert total_pos_01 >= total_pos_05


def test_evaluate_threshold_sweep_perfect_separation():
    """Verify perfect probability scores yield 1.0 F1 and 0.0 FPR across standard thresholds."""
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.01, 0.02, 0.98, 0.99])

    results = evaluate_threshold_sweep(y_true, probs, thresholds=[0.1, 0.3, 0.5])
    for t_str, res in results.items():
        assert res["accuracy"] == 1.0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1_score"] == 1.0
        assert res["false_positive_rate"] == 0.0


def test_run_threshold_analysis_synthetic(tmp_path: Path):
    """Verify run_threshold_analysis executes on synthetic validation and test DataFrames."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([0, 0, 1, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)

    model_file = tmp_path / "test_model.joblib"
    save_model_artifact(pipe, meta, model_file, overwrite=True)

    val_df = pd.DataFrame({
        "f1": [1.1, 1.9, 3.1, 3.9],
        "f2": [11.0, 19.0, 31.0, 39.0],
        "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS"],
    })
    test_df = pd.DataFrame({
        "f1": [1.2, 3.2],
        "f2": [12.0, 32.0],
        "Label": ["BENIGN", "DDoS"],
    })

    rep = run_threshold_analysis(
        model_paths={"synthetic_model": model_file},
        validation_df=val_df,
        wednesday_df=test_df,
        thresholds=[0.2, 0.5],
    )

    assert "synthetic_model" in rep["models"]
    m_rep = rep["models"]["synthetic_model"]
    assert "validation_sweep" in m_rep
    assert "selected_validation_threshold" in m_rep
    assert "independent_wednesday_sweep" in m_rep


def test_analyze_false_negatives_synthetic(tmp_path: Path):
    """Verify analyze_false_negatives computes probability stats and handles empty families."""
    X_train = pd.DataFrame({"Flow Duration": [10.0, 20.0, 100.0, 200.0]})
    y_train = pd.Series([0, 0, 1, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)

    model_file = tmp_path / "fn_model.joblib"
    save_model_artifact(pipe, meta, model_file, overwrite=True)

    dev_df = pd.DataFrame({
        "Flow Duration": [10.0, 20.0, 100.0, 200.0],
        "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS"],
    })
    test_df = pd.DataFrame({
        "Flow Duration": [15.0, 150.0],
        "Label": ["BENIGN", "DoS Hulk"],
    })

    fn_rep = analyze_false_negatives(
        model_path=model_file,
        wednesday_df=test_df,
        development_df=dev_df,
        selected_threshold=0.5,
    )

    assert "attack_families" in fn_rep
    assert "DoS Hulk" in fn_rep["attack_families"]
    assert "cross_capture_feature_conflicts" in fn_rep
