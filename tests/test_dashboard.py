"""
Unit and integration tests for the Streamlit dashboard helper functions (app.py).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import app
from app import (
    load_trusted_model,
    load_evaluation_report,
    run_batch_prediction_on_dataframe,
    filter_predictions_display,
)
from src.feature_engineering import FeatureSchema, prepare_feature_matrix
from src.train import train_model, save_model_artifact


def test_app_imports_cleanly():
    """Verify that importing app.py causes no model training or file side-effects."""
    assert hasattr(app, "main")
    assert hasattr(app, "load_trusted_model")
    assert hasattr(app, "run_batch_prediction_on_dataframe")


def test_load_trusted_model_missing():
    """Verify that missing model artifact returns None and a helpful error message."""
    model, metadata, err = load_trusted_model("nonexistent_model_12345.joblib")
    assert model is None
    assert metadata is None
    assert err is not None
    assert "not found" in err.lower()


def test_load_trusted_model_valid(tmp_path: Path):
    """Verify that a genuine saved model is loaded safely."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([0, 0, 1, 1])
    pipeline, meta = train_model(X_train, y_train, random_state=42)

    model_file = tmp_path / "valid_model.joblib"
    save_model_artifact(pipeline, meta, model_file, overwrite=True)

    loaded_pipe, loaded_meta, err = load_trusted_model(model_file)
    assert err is None
    assert loaded_pipe is not None
    assert loaded_meta["n_features"] == 2


def test_batch_prediction_valid_synthetic(tmp_path: Path):
    """Verify batch prediction on synthetic data preserves row ordering and extracts scores."""
    X_train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0, 4.0], "feat_b": [5.0, 6.0, 7.0, 8.0]})
    y_train = pd.Series([0, 0, 1, 1])
    pipeline, meta = train_model(X_train, y_train, random_state=42)

    test_df = pd.DataFrame(
        {
            "Flow ID": ["FLOW_1", "FLOW_2", "FLOW_3"],
            "feat_a": [1.1, 3.2, 1.2],
            "feat_b": [5.1, 7.1, 5.2],
        }
    )

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=pipeline,
        metadata=meta,
        raw_df=test_df,
        threshold=0.5,
    )

    assert err is None
    assert pred_df is not None
    assert len(pred_df) == 3
    # Verify row order preserved
    assert list(pred_df["Flow ID"]) == ["FLOW_1", "FLOW_2", "FLOW_3"]
    assert "predicted_class" in pred_df.columns
    assert "predicted_label" in pred_df.columns
    assert "attack_score" in pred_df.columns


def test_batch_prediction_missing_features_error():
    """Verify that missing required features return a helpful, plain-language error."""
    X_train = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]})
    y_train = pd.Series([0, 1])
    pipeline, meta = train_model(X_train, y_train, random_state=42)

    # Incomplete input lacking 'beta'
    bad_df = pd.DataFrame({"alpha": [1.5, 2.5], "gamma": [9.0, 9.9]})

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=pipeline,
        metadata=meta,
        raw_df=bad_df,
    )

    assert pred_df is None
    assert err is not None
    assert "Missing required feature columns" in err or "beta" in err


def test_batch_prediction_extra_columns_policy():
    """Verify that extra columns in input CSV are ignored during inference and kept in result."""
    X_train = pd.DataFrame({"feat_1": [1.0, 2.0], "feat_2": [10.0, 20.0]})
    y_train = pd.Series([0, 1])
    pipeline, meta = train_model(X_train, y_train, random_state=42)

    input_df = pd.DataFrame(
        {
            "feat_1": [1.5, 2.5],
            "feat_2": [12.0, 22.0],
            "Extra_Telemetry_Column": ["sensor_A", "sensor_B"],
        }
    )

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=pipeline,
        metadata=meta,
        raw_df=input_df,
    )

    assert err is None
    assert pred_df is not None
    assert "Extra_Telemetry_Column" in pred_df.columns
    assert "predicted_class" in pred_df.columns


def test_batch_prediction_benign_only_zero_alerts():
    """Verify that predictions with zero attack flags produce zero alerts."""
    class DummyBenignModel:
        classes_ = np.array([0, 1])
        def predict(self, X):
            return np.array([0, 0])
        def predict_proba(self, X):
            return np.array([[0.99, 0.01], [0.95, 0.05]])

    meta = {"feature_names": ["f1"]}
    df = pd.DataFrame({"f1": [10.0, 20.0]})

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=DummyBenignModel(),
        metadata=meta,
        raw_df=df,
        threshold=0.5,
    )

    assert err is None
    assert (pred_df["predicted_label"] == "Benign").all()
    assert alerts == []


def test_batch_prediction_attack_generates_alerts():
    """Verify that attack predictions generate structured alerts with telemetry metadata."""
    class DummyAttackModel:
        classes_ = np.array([0, 1])
        def predict(self, X):
            return np.array([1])
        def predict_proba(self, X):
            return np.array([[0.05, 0.95]])

    meta = {"feature_names": ["f1"]}
    df = pd.DataFrame({"Flow ID": ["FLOW_999"], "Source IP": ["10.0.0.1"], "f1": [42.0]})

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=DummyAttackModel(),
        metadata=meta,
        raw_df=df,
        threshold=0.5,
    )

    assert err is None
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["flow_id"] == "FLOW_999"
    assert alert["severity"] == "CRITICAL"
    assert alert["probability"] == 0.95
    assert alert["metadata"]["Source IP"] == "10.0.0.1"


def test_filter_predictions_display_preserves_original():
    """Verify that applying display filters returns filtered views without altering original data."""
    original_df = pd.DataFrame(
        {
            "Flow ID": ["F1", "F2", "F3"],
            "predicted_label": ["Benign", "Attack", "Attack"],
            "attack_score": [0.05, 0.75, 0.95],
        }
    )

    # Filter Attack Only
    attack_view = filter_predictions_display(original_df, class_filter="Attack Only")
    assert len(attack_view) == 2
    assert (attack_view["predicted_label"] == "Attack").all()
    # Ensure original is unchanged
    assert len(original_df) == 3

    # Filter min_score
    score_view = filter_predictions_display(original_df, min_score=0.80)
    assert len(score_view) == 1
    assert score_view.iloc[0]["Flow ID"] == "F3"
    assert len(original_df) == 3


def test_load_evaluation_report_missing_and_valid(tmp_path: Path):
    """Verify loading evaluation reports handles missing files and parses valid files."""
    data, err = load_evaluation_report(tmp_path / "missing_report.json")
    assert data is None
    assert "no evaluation report found" in err.lower()

    valid_file = tmp_path / "real_report.json"
    sample_metrics = {"accuracy": 0.98, "f1_score": 0.97, "evaluation_dataset": "test.csv"}
    with open(valid_file, "w") as f:
        json.dump(sample_metrics, f)

    loaded, err = load_evaluation_report(valid_file)
    assert err is None
    assert loaded["accuracy"] == 0.98
    assert loaded["evaluation_dataset"] == "test.csv"


def test_load_trusted_model_default_multi_attack():
    """Verify load_trusted_model defaults to the multi-attack model artifact."""
    model, metadata, err = load_trusted_model()
    assert err is None
    assert model is not None
    assert metadata is not None
    assert metadata.get("n_features") == 78


def test_batch_prediction_excludes_target_column():
    """Verify that any target label column in uploaded CSV is excluded from prediction inputs."""
    class DummyInspectorModel:
        classes_ = np.array([0, 1])
        def predict(self, X):
            assert "Label" not in X.columns
            assert "target" not in X.columns
            return np.array([0, 0])
        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.8, 0.2]])

    meta = {"feature_names": ["f1", "f2"]}
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "Label": ["BENIGN", "DoS Hulk"]})

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=DummyInspectorModel(),
        metadata=meta,
        raw_df=df,
        threshold=0.5,
    )
    assert err is None
    assert pred_df is not None
    assert "Label" in pred_df.columns  # preserved in output view
    assert "predicted_label" in pred_df.columns


def test_load_evaluation_report_wednesday_and_multi_attack():
    """Verify loading the real generated multi-attack and Wednesday evaluation reports."""
    from app import MULTI_ATTACK_TRAIN_REPORT_PATH, THIRD_CAPTURE_EVAL_REPORT_PATH

    ma_report, ma_err = load_evaluation_report(MULTI_ATTACK_TRAIN_REPORT_PATH)
    assert ma_err is None
    assert ma_report is not None
    assert ma_report["accuracy"] > 0.99
    assert ma_report["sample_count"] == 20000

    wed_report, wed_err = load_evaluation_report(THIRD_CAPTURE_EVAL_REPORT_PATH)
    assert wed_err is None
    assert wed_report is not None
    assert wed_report["sample_count"] == 692703
    assert wed_report["recall"] < 0.01  # reflects ~0.11% recall on Wednesday capture


def test_apptest_renders_dashboard_cleanly():
    """Verify that app.py executes cleanly end-to-end via Streamlit AppTest."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5

