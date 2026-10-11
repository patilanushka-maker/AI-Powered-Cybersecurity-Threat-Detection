"""
Automated unit tests for src/train.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.train import (
    build_training_pipeline,
    train_model,
    save_model_artifact,
    load_model_artifact,
)


def test_train_model_synthetic_binary():
    """Test 9 & 11: Successful training and prediction on synthetic binary dataset."""
    X_train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5],
            "f2": [10.0, 12.0, 11.0, 80.0, 90.0, 85.0],
        }
    )
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    pipeline, metadata = train_model(X_train, y_train, random_state=42)

    assert pipeline is not None
    assert metadata["model_type"] == "RandomForestClassifier"
    assert metadata["classes"] == [0, 1]
    assert metadata["n_train_samples"] == 6

    # Verify prediction works
    preds = pipeline.predict(X_train)
    assert len(preds) == 6
    assert list(preds) == [0, 0, 0, 1, 1, 1]


def test_imputation_learned_from_train_only():
    """
    Test 10: Missing-value imputation statistics learned strictly from training data.
    Train data has feature with values [10, 20, 30] -> median is 20.
    Test data with missing value should be imputed with 20, not test data values.
    """
    X_train = pd.DataFrame({"val": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    pipeline, _ = train_model(X_train, y_train, imputer_strategy="median")

    # Inspect fitted imputer step
    imputer = pipeline.named_steps["imputer"]
    assert imputer.statistics_[0] == 20.0

    # Transform test set containing NaN and an extreme value [1000, NaN]
    X_test = pd.DataFrame({"val": [np.nan, 1000.0]})
    imputed_test = pipeline.named_steps["imputer"].transform(X_test)
    assert imputed_test[0, 0] == 20.0  # Filled with train median


def test_configurable_estimator_logistic_regression():
    """Test configurable estimator (Logistic Regression baseline)."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 8.0, 9.0]})
    y_train = pd.Series([0, 0, 1, 1])

    clf = LogisticRegression(random_state=42)
    pipeline, metadata = train_model(X_train, y_train, estimator=clf)

    assert metadata["model_type"] == "LogisticRegression"
    assert "classifier" in pipeline.named_steps


def test_save_and_reload_model_artifact(tmp_path: Path):
    """Test 12: Save and reload a model artifact using a temporary test directory."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 8.0, 9.0]})
    y_train = pd.Series([0, 0, 1, 1])

    pipeline, metadata = train_model(X_train, y_train, random_state=42)
    artifact_path = tmp_path / "test_model.joblib"

    # Save artifact
    saved_path = save_model_artifact(
        pipeline=pipeline,
        metadata=metadata,
        output_path=artifact_path,
        overwrite=False,
    )
    assert saved_path.exists()

    # Prevent accidental overwrite without overwrite=True
    with pytest.raises(FileExistsError):
        save_model_artifact(pipeline, metadata, artifact_path, overwrite=False)

    # Reload artifact and test consistency
    loaded_pipe, loaded_meta = load_model_artifact(artifact_path)
    assert loaded_meta["model_type"] == metadata["model_type"]
    assert list(loaded_pipe.predict(X_train)) == list(pipeline.predict(X_train))


def test_train_model_unsuitable_data_errors():
    """Test that empty or 1-class data raises informative ValueError."""
    # Empty dataset
    with pytest.raises(ValueError):
        train_model(pd.DataFrame(), pd.Series(dtype=int))

    # Single class dataset
    X = pd.DataFrame({"f": [1, 2, 3]})
    y = pd.Series([0, 0, 0])
    with pytest.raises(ValueError) as exc:
        train_model(X, y)
    assert "must contain at least 2 distinct classes" in str(exc.value)
