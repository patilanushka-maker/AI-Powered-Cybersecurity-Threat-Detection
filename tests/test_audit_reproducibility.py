"""
Automated Unit Tests for Milestone 18 Independent Reproducibility and Metric Audit.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd


def test_confusion_matrix_arithmetic_assertions():
    """Verify confusion matrix arithmetic identities hold exactly for any classification output."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 1])

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total_attacks = int(np.sum(y_true == 1))
    total_benign = int(np.sum(y_true == 0))
    total_rows = len(y_true)

    assert tp + fn == total_attacks
    assert tn + fp == total_benign
    assert tp + fp + tn + fn == total_rows


def test_positive_class_index_resolution():
    """Verify positive class index is resolved dynamically from classes_."""
    class MockEstimator:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            # Column 0 is class 0, Column 1 is class 1
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    est = MockEstimator()
    pos_idx = list(est.classes_).index(1)
    assert pos_idx == 1

    # Also test reversed classes order [1, 0]
    class MockReversedEstimator:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.9, 0.1]])

    rev_est = MockReversedEstimator()
    rev_pos_idx = list(rev_est.classes_).index(1)
    assert rev_pos_idx == 0


def test_family_count_reconciliation():
    """Verify per-family attack counts and true positives sum up to total attack counts."""
    labels = np.array(["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk", "DoS slowloris", "Heartbleed"])
    y_true_binary = np.array([0, 0, 1, 1, 1, 1])
    y_pred_binary = np.array([0, 0, 1, 0, 1, 0])

    families = ["DoS Hulk", "DoS slowloris", "Heartbleed"]
    tot_rows = 0
    tot_tp = 0

    for fam in families:
        mask = labels == fam
        n = int(np.sum(mask))
        tp = int(np.sum((y_pred_binary[mask] == 1)))
        tot_rows += n
        tot_tp += tp

    overall_attacks = int(np.sum(y_true_binary == 1))
    overall_tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))

    assert tot_rows == overall_attacks
    assert tot_tp == overall_tp


def test_feature_vector_exact_equality():
    """Verify value-by-value exact matching identifies identical rows and catches differences."""
    df1 = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [10.0, 20.0, 30.0]})
    df2 = pd.DataFrame({"f1": [2.0, 4.0], "f2": [20.0, 40.0]})

    merged = pd.merge(df1, df2, on=["f1", "f2"])
    assert len(merged) == 1
    assert merged.iloc[0]["f1"] == 2.0
    assert merged.iloc[0]["f2"] == 20.0


def test_milestone18_audit_reports_exist_and_pass():
    """Verify Milestone 18 audit reports exist and confirm all audit passes."""
    audit_json = Path("outputs/reports/milestone18_reproducibility_audit.json")
    audit_md = Path("outputs/reports/milestone18_audit_summary.md")

    assert audit_json.exists(), "milestone18_reproducibility_audit.json missing"
    assert audit_md.exists(), "milestone18_audit_summary.md missing"

    with open(audit_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["overall_audit_status"] == "PASSED_WITH_DOCUMENTED_LIMITATIONS"
    assert data["confusion_matrix_arithmetic_assertions"]["status"] == "ALL_ASSERTIONS_PASSED"
    assert data["exact_feature_collision_audit"]["wednesday_dos_hulk_rows"] == 4187
    assert data["exact_feature_collision_audit"]["unique_conflicting_feature_vectors"] == 36
    assert data["threshold_selection_audit"]["status"] == "VERIFIED_INDEPENDENT"
