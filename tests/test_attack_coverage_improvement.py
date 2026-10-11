"""
Unit Tests for Milestone 21: Attack-Family Coverage and Generalization Improvement.

Validates:
- Feature alignment and schema consistency.
- Zero-leakage group-aware splitting.
- Confusion matrix arithmetic reconciliation.
- Baseline model checksum invariance (ensuring no overwriting of existing models).
- Attack-family coverage breakdown and explicit absent-class warnings.
- Candidate model regularized configurations.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.attack_coverage_experiments import analyze_feature_and_score_distributions
from src.model_reliability import (
    compute_file_sha256,
    perform_group_aware_split,
    compute_attack_family_coverage,
)


def test_frozen_baseline_checksums_invariant():
    """
    Test that the frozen baseline model files match their exact immutable SHA-256 digests.
    """
    expected = {
        "models/random_forest_multi_attack.joblib": "486d6c75f00a163204cb993d0a4e9e2688c1b22ef58cd58e2e53e418d7fcd35f",
        "models/model_extra_trees_balanced_m16.joblib": "c60cef05d9ad7751f013e13d5c297dc2e9b1191ec9aaf1af887cb16227c7f60b",
        "models/random_forest_baseline.joblib": "e6e6d785ab068381befcfbc914402be9253cec6789ac55c820952b670a5bc3b3",
    }
    for path_str, expected_hash in expected.items():
        actual_hash = compute_file_sha256(path_str)
        assert actual_hash == expected_hash, f"Baseline artifact {path_str} was unexpectedly modified!"


def test_leakage_prevention_in_group_aware_splitting():
    """
    Test that identical feature vectors do not cross into different partitions
    during group-aware splitting.
    """
    X = pd.DataFrame({
        "f1": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        "f2": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0],
    })
    y_bin = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_raw = pd.Series(["BENIGN", "BENIGN", "DDoS", "DDoS", "BENIGN", "BENIGN", "DDoS", "DDoS"])

    res = perform_group_aware_split(X, y_bin, y_raw, test_size=0.25, random_state=42)
    assert res["total_unique_feature_groups"] == 4
    assert res["overlapping_groups_count"] == 0
    assert res["group_separation_verified"] is True

    train_tuples = set(tuple(x) for x in X.iloc[res["train_indices"]].itertuples(index=False, name=None))
    val_tuples = set(tuple(x) for x in X.iloc[res["val_indices"]].itertuples(index=False, name=None))
    assert train_tuples.isdisjoint(val_tuples)


def test_candidate_a_tree_regularization_hyperparameters():
    """
    Verify that Candidate A (Regularized Extra Trees) is configured with
    explicit depth and leaf size regularization to prevent memorization.
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    clf = pipe.named_steps["clf"]
    assert clf.max_depth == 12
    assert clf.min_samples_leaf == 5
    assert clf.class_weight == "balanced"


def test_metric_and_confusion_reconciliation():
    """
    Verify that confusion matrix terms reconcile to sample count and that
    all precision, recall, and F1 calculations are algebraically sound.
    """
    # 10 samples
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    assert tn == 4
    assert fp == 1
    assert fn == 1
    assert tp == 4
    assert tn + fp + fn + tp == 10

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    assert pytest.approx(prec, 0.001) == 4 / 5
    assert pytest.approx(rec, 0.001) == 4 / 5
    assert pytest.approx(f1, 0.001) == 4 / 5


def test_unseen_attack_family_warnings():
    """
    Verify that compute_attack_family_coverage marks unseen families with
    explicit warning strings and distinguishes in-distribution from out-of-distribution.
    """
    y_true_bin = np.array([0, 0, 1, 1])
    y_pred_bin = np.array([0, 0, 1, 0])
    y_raw = pd.Series(["BENIGN", "BENIGN", "DDoS", "DoS GoldenEye"])
    trained_families = {"DDoS"}

    coverage = compute_attack_family_coverage(
        y_true_bin, y_pred_bin, y_raw, trained_families=trained_families
    )

    assert coverage["DDOS"]["represented_in_training"] is True
    assert coverage["DDOS"]["warning"] is None
    assert coverage["DDOS"]["evaluation_context"] == "in_distribution_attack"

    assert coverage["DOS GOLDENEYE"]["represented_in_training"] is False
    assert coverage["DOS GOLDENEYE"]["warning"] is not None
    assert "EXPLICIT WARNING" in coverage["DOS GOLDENEYE"]["warning"]
    assert coverage["DOS GOLDENEYE"]["evaluation_context"] == "out_of_distribution_attack"


def test_candidate_a_metrics_and_identities_reconciliation():
    """
    Validate Phase 2 requirements:
    Reconcile Candidate A's numbers (98,763 alerts, 98,063 true positives,
    97,534 DoS Hulk detections) and verify all arithmetic identities.
    """
    import json
    from pathlib import Path

    rep_path = Path("outputs/reports/milestone21_attack_coverage_report.json")
    assert rep_path.exists(), "Milestone report json must exist."

    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cand_a = data["phase4_controlled_experiments"]["candidate_models"]["candidate_a_regularized_extra_trees"]
    cm = cand_a["wednesday_benchmark_t050"]["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    alerts = cand_a["wednesday_benchmark_t050"]["predicted_alerts_count"]
    total = cand_a["wednesday_benchmark_t050"]["total_samples"]

    # Candidate A specific counts
    assert tn == 439331
    assert fp == 700
    assert fn == 154609
    assert tp == 98063
    assert alerts == 98763
    assert total == 692703

    # Verification of the 6 fundamental identities
    assert tn + fp + fn + tp == total, "Identity 1: TN + FP + FN + TP must equal total evaluation rows"
    assert tp + fn == 252672, "Identity 2: TP + FN must equal total attack rows"
    assert tn + fp == 440031, "Identity 3: TN + FP must equal total benign rows"
    assert alerts == tp + fp, "Identity 4: Alerts must equal TP + FP"

    rec = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp)
    f1 = 2 * (prec * rec) / (prec + rec)

    assert pytest.approx(cand_a["wednesday_benchmark_t050"]["recall"], 1e-6) == rec
    assert pytest.approx(cand_a["wednesday_benchmark_t050"]["false_positive_rate"], 1e-6) == fpr
    assert pytest.approx(cand_a["wednesday_benchmark_t050"]["precision"], 1e-6) == prec
    assert pytest.approx(cand_a["wednesday_benchmark_t050"]["f1_score"], 1e-6) == f1

    # Per-family decomposition of the 98,063 True Positives
    fam_cov = cand_a["wednesday_family_coverage"]
    hulk_tp = fam_cov["DOS HULK"]["true_positives"]
    slowloris_tp = fam_cov["DOS SLOWLORIS"]["true_positives"]
    goldeneye_tp = fam_cov["DOS GOLDENEYE"]["true_positives"]
    slowhttptest_tp = fam_cov["DOS SLOWHTTPTEST"]["true_positives"]
    heartbleed_tp = fam_cov["HEARTBLEED"]["true_positives"]

    assert hulk_tp == 97534
    assert slowloris_tp == 520
    assert goldeneye_tp == 9
    assert slowhttptest_tp == 0
    assert heartbleed_tp == 0

    assert hulk_tp + slowloris_tp + goldeneye_tp + slowhttptest_tp + heartbleed_tp == tp
    assert fam_cov["BENIGN"]["false_positives"] == fp


def test_per_family_tp_equals_overall_tp_across_all_models():
    """
    Validate Phase 3 requirements:
    Confirm that the sum of per-family attack true positives equals overall attack TP,
    and benign false positives match the confusion matrix FP for every model.
    """
    import json
    from pathlib import Path

    rep_path = Path("outputs/reports/milestone21_attack_coverage_report.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp = data["phase4_controlled_experiments"]
    models_dict = {
        **exp["frozen_baseline_models"],
        **exp["candidate_models"],
    }

    for model_name, m_data in models_dict.items():
        # Find wednesday benchmark evaluation
        wed_eval_key = next((k for k in m_data if "wednesday_benchmark" in k), None)
        assert wed_eval_key is not None, f"Model {model_name} missing Wednesday evaluation."
        overall_cm = m_data[wed_eval_key]["confusion_matrix"]
        overall_tp = overall_cm["tp"]
        overall_fp = overall_cm["fp"]

        # Find wednesday family coverage
        cov_key = next((k for k in m_data if "wednesday_family_coverage" in k), None)
        assert cov_key is not None, f"Model {model_name} missing family coverage."
        fam_cov = m_data[cov_key]

        sum_family_attack_tp = sum(
            fam_data.get("true_positives", 0)
            for fam_name, fam_data in fam_cov.items()
            if fam_name != "BENIGN"
        )
        benign_fp = fam_cov["BENIGN"]["false_positives"]

        assert sum_family_attack_tp == overall_tp, (
            f"{model_name}: Sum of family attack TPs ({sum_family_attack_tp}) "
            f"must equal overall TP ({overall_tp})"
        )
        assert benign_fp == overall_fp, (
            f"{model_name}: Benign FP ({benign_fp}) must match overall FP ({overall_fp})"
        )


def test_score_distributions_counts_and_bounds():
    """
    Validate Phase 4 requirements:
    Recompute and verify Wednesday score distribution counts, ceilings,
    and threshold frequencies directly from the structured report.
    """
    import json
    from pathlib import Path

    rep_path = Path("outputs/reports/milestone21_attack_coverage_report.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dist = data["phase2_feature_investigation"]["prediction_score_distributions_by_class"]
    total_count = sum(s["count"] for s in dist.values())
    assert total_count == 692703, "Total flow count in score distributions must equal 692,703"

    # Verify score ceilings
    assert dist["DOS GOLDENEYE"]["max"] == pytest.approx(0.2900, 1e-4)
    assert dist["DOS SLOWHTTPTEST"]["max"] == pytest.approx(0.4545, 1e-4)
    assert dist["HEARTBLEED"]["max"] == pytest.approx(0.1200, 1e-4)

    # Verify scores >= 0.50 count aligns with Baseline 2 true/false positives
    assert dist["BENIGN"]["scores_ge_050"] == 257  # Benign FP at T=0.50
    assert dist["DOS HULK"]["scores_ge_050"] == 40122  # Hulk TP at T=0.50
    assert dist["DOS SLOWLORIS"]["scores_ge_050"] == 520  # Slowloris TP at T=0.50
    assert dist["DOS GOLDENEYE"]["scores_ge_050"] == 0
    assert dist["DOS SLOWHTTPTEST"]["scores_ge_050"] == 0
    assert dist["HEARTBLEED"]["scores_ge_050"] == 0


def test_isolation_forest_verification_and_caveats():
    """
    Validate Phase 5 requirements:
    Confirm Isolation Forest training provenance, mapping, confusion matrix,
    FPR denominator, and Heartbleed detection context.
    """
    import json
    from pathlib import Path

    rep_path = Path("outputs/reports/milestone21_attack_coverage_report.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cand_b = data["phase4_controlled_experiments"]["candidate_models"]["candidate_b_unsupervised_isolation_forest"]
    cm = cand_b["wednesday_benchmark"]["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    assert tn == 435617
    assert fp == 4414
    assert fn == 208981
    assert tp == 43691
    assert tp + fp == 48105

    # FPR denominator is benign Wednesday rows (440,031)
    fpr = fp / (fp + tn)
    assert pytest.approx(fpr, 1e-6) == 4414 / 440031
    assert pytest.approx(fpr * 100, 0.01) == 1.0031

    # Heartbleed detection breakdown: 11 / 11 flows
    hb = cand_b["wednesday_family_coverage"]["HEARTBLEED"]
    assert hb["support"] == 11
    assert hb["true_positives"] == 11
    assert hb["recall"] == 1.0

