"""
Model Training Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Constructing leakage-free Scikit-learn pipelines combining imputation and estimators
- Training Random Forest classifiers (and configurable baselines like Logistic Regression)
- Recording model metadata, feature schemas, and hyperparameters
- Safe serialization and deserialization of fitted model pipelines to models/
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.feature_engineering import FeatureSchema

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "random_forest_baseline.joblib"


def build_training_pipeline(
    estimator: Optional[BaseEstimator] = None,
    imputer_strategy: str = "median",
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> Pipeline:
    """
    Construct a unified Scikit-learn Pipeline encapsulating numerical imputation
    and the classifier estimator.

    Parameters
    ----------
    estimator : Optional[BaseEstimator], default None
        Scikit-learn classifier. If None, defaults to RandomForestClassifier
        with balanced class weights.
    imputer_strategy : str, default 'median'
        Imputation strategy for missing values ('median', 'mean', or 'most_frequent').
        Fitted exclusively on training data during pipeline.fit().
    random_state : int, default 42
        Seed for reproducibility when creating default estimator.
    n_estimators : int, default 100
        Number of trees if default RandomForestClassifier is built.
    max_depth : Optional[int], default None
        Maximum tree depth if default RandomForestClassifier is built.

    Returns
    -------
    Pipeline
        Unfitted Scikit-learn Pipeline with 'imputer' and 'classifier' steps.
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("classifier", estimator),
        ]
    )
    return pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator: Optional[BaseEstimator] = None,
    imputer_strategy: str = "median",
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    feature_schema: Optional[FeatureSchema] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a complete Scikit-learn classification pipeline on partitioned training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels (e.g. 0 for Benign, 1 for Attack).
    estimator : Optional[BaseEstimator], default None
        Estimator to train. Defaults to RandomForestClassifier.
    imputer_strategy : str, default 'median'
        Missing value strategy for SimpleImputer.
    random_state : int, default 42
        Seed for reproducibility.
    n_estimators : int, default 100
        Number of trees if default RandomForestClassifier is used.
    max_depth : Optional[int], default None
        Maximum tree depth if default RandomForestClassifier is used.
    feature_schema : Optional[FeatureSchema], default None
        Feature schema metadata. If None, created automatically from X_train.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        (fitted_pipeline, metadata_dictionary)

    Raises
    ------
    ValueError
        If inputs are empty, contain fewer than 2 classes, or have length mismatch.
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot train model on empty training data.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Length mismatch: X_train has {len(X_train)} rows, y_train has {len(y_train)} rows."
        )

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Training data must contain at least 2 distinct classes. Found: {list(unique_classes)}"
        )

    # Build and fit pipeline
    pipeline = build_training_pipeline(
        estimator=estimator,
        imputer_strategy=imputer_strategy,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    pipeline.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    classifier_step = pipeline.named_steps["classifier"]
    classes = [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in classifier_step.classes_]

    metadata: Dict[str, Any] = {
        "model_type": classifier_step.__class__.__name__,
        "training_timestamp": datetime.utcnow().isoformat(),
        "n_train_samples": len(X_train),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "classes": classes,
        "imputer_strategy": imputer_strategy,
        "random_state": random_state,
        "estimator_params": classifier_step.get_params(),
        "feature_schema": feature_schema.to_dict() if feature_schema else None,
    }

    return pipeline, metadata


def save_model_artifact(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    overwrite: bool = False,
) -> Path:
    """
    Safely serialize a trained pipeline and its metadata to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted Scikit-learn Pipeline.
    metadata : Dict[str, Any]
        Model metadata dictionary.
    output_path : Union[str, Path], default 'models/random_forest_baseline.joblib'
        Target file path.
    overwrite : bool, default False
        If False and file exists, raises FileExistsError.

    Returns
    -------
    Path
        Path to saved artifact.

    Raises
    ------
    FileExistsError
        If destination exists and overwrite=False.
    """
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Model artifact already exists at '{path}'. Pass overwrite=True to overwrite."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "metadata": metadata,
    }
    joblib.dump(payload, path)
    return path


def load_model_artifact(
    artifact_path: Union[str, Path]
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Load a serialized model artifact and verify required keys.

    CAUTION: Deserializing Joblib/pickle artifacts executes arbitrary code.
    Only load trusted model artifacts generated by your own training pipeline.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to .joblib file.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        (pipeline, metadata)

    Raises
    ------
    FileNotFoundError
        If artifact_path does not exist.
    ValueError
        If artifact format is invalid or missing required keys.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    payload = joblib.load(path)
    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        raise ValueError(
            f"Invalid model artifact structure at '{path}'. Expected dict with 'pipeline' and 'metadata'."
        )

    return payload["pipeline"], payload["metadata"]
