"""
Automated unit tests for src/predict.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.predict import load_model_for_inference, predict_network_flows
from src.train import train_model, save_model_artifact
from src.feature_engineering import FeatureSchema


def test_load_model_missing_file_raises_error():
    """Verify that attempting to load a missing model file raises FileNotFoundError."""
    dummy_path = Path("models/non_existent.joblib")
    with pytest.raises(FileNotFoundError):
        load_model_for_inference(dummy_path)


def test_predict_network_flows_empty_df():
    """Verify that passing an empty DataFrame raises ValueError."""
    class DummyModel:
        pass

    with pytest.raises(ValueError):
        predict_network_flows(DummyModel(), pd.DataFrame())


def test_predict_network_flows_with_pipeline_and_schema(tmp_path: Path):
    """Verify end-to-end prediction, probabilities in [0, 1], and schema validation."""
    X_train = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 10.0, 11.0],
            "feat_b": [0.5, 0.8, 8.5, 9.0],
        }
    )
    y_train = pd.Series([0, 0, 1, 1])

    schema = FeatureSchema(
        feature_names=["feat_a", "feat_b"],
        numeric_features=["feat_a", "feat_b"],
    )

    pipeline, meta = train_model(X_train, y_train, feature_schema=schema)

    model_file = tmp_path / "model.joblib"
    save_model_artifact(pipeline, meta, output_path=model_file)

    loaded_pipe, loaded_meta = load_model_for_inference(model_file)

    # Inference data with extra column and reordered columns
    new_data = pd.DataFrame(
        {
            "extra_col": ["ignore_me", "ignore_me"],
            "feat_b": [0.6, 8.8],
            "feat_a": [1.5, 10.5],
        }
    )

    preds, probs = predict_network_flows(
        loaded_pipe, new_data, feature_schema=schema, pos_label=1
    )

    assert len(preds) == 2
    assert list(preds) == [0, 1]
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_network_flows_reversed_classes_order():
    """
    Test Phase 2.2: Controlled estimator with explicitly reversed class ordering [1, 0].
    Verifies that the positive class 1 probability is chosen by index lookup on classes_,
    NOT by assuming index 1.
    """
    class ReversedClassModel:
        classes_ = np.array([1, 0])  # Index 0 is class 1 (Attack), Index 1 is class 0 (Benign)

        def predict(self, X):
            return np.array([1, 0])

        def predict_proba(self, X):
            # Row 0: P(class 1)=0.85, P(class 0)=0.15
            # Row 1: P(class 1)=0.10, P(class 0)=0.90
            return np.array([[0.85, 0.15], [0.10, 0.90]])

    df = pd.DataFrame({"f1": [10, 20]})
    preds, probs = predict_network_flows(ReversedClassModel(), df, pos_label=1)

    assert list(preds) == [1, 0]
    # Pos class 1 must be 0.85 and 0.10, NOT 0.15 and 0.90!
    assert probs[0] == 0.85
    assert probs[1] == 0.10


def test_predict_network_flows_missing_proba_support_raises():
    """Verify ValueError when estimator does not implement predict_proba."""
    class NoProbaModel:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1])

    df = pd.DataFrame({"f1": [1, 2]})
    with pytest.raises(ValueError) as exc:
        predict_network_flows(NoProbaModel(), df)
    assert "predict_proba missing" in str(exc.value)


def test_predict_network_flows_missing_pos_label_raises():
    """Verify ValueError when positive class is absent from estimator classes_."""
    class MissingClassModel:
        classes_ = np.array([0, 2])  # Class 1 is missing

        def predict(self, X):
            return np.array([0, 2])

        def predict_proba(self, X):
            return np.array([[0.7, 0.3], [0.4, 0.6]])

    df = pd.DataFrame({"f1": [1, 2]})
    with pytest.raises(ValueError) as exc:
        predict_network_flows(MissingClassModel(), df, pos_label=1)
    assert "Positive class '1' not found" in str(exc.value)


def test_predict_network_flows_non_finite_probs_raises():
    """Verify ValueError when estimator outputs NaN or Inf probabilities."""
    class NonFiniteProbaModel:
        classes_ = np.array([0, 1])

        def predict(self, X):
            return np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.5, np.nan], [0.2, 0.8]])

    df = pd.DataFrame({"f1": [1, 2]})
    with pytest.raises(ValueError) as exc:
        predict_network_flows(NonFiniteProbaModel(), df, pos_label=1)
    assert "non-finite" in str(exc.value).lower()
