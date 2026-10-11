"""
Inference / Prediction Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Loading verified model artifacts for offline scoring
- Aligning and validating incoming flow features against FeatureSchema
- Generating binary classifications and positive-class attack probability scores
- Strict positive-class resolution and finite probability bounds [0.0, 1.0]
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from src.feature_engineering import FeatureSchema
from src.train import load_model_artifact


def load_model_for_inference(
    model_path: Union[str, Path]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Safely load a serialized model artifact and its metadata for inference.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the serialized .joblib model file.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        (loaded_pipeline, metadata_dictionary)

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    return load_model_artifact(path)


def predict_network_flows(
    model: Any,
    flow_data: pd.DataFrame,
    feature_schema: Optional[FeatureSchema] = None,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate threat classifications and probability estimates for input flow records.

    Parameters
    ----------
    model : Any
        Fitted model or pipeline.
    flow_data : pd.DataFrame
        Unseen network flow feature DataFrame.
    feature_schema : Optional[FeatureSchema], default None
        Optional schema to validate columns, types, and alignment.
    pos_label : int, default 1
        Positive class identifier for probability extraction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (predicted_labels, attack_confidence_probabilities)

    Raises
    ------
    ValueError
        If flow_data is empty, fails schema validation, estimator does not support
        probabilities, positive class is absent, or probabilities are non-finite.
    """
    if flow_data.empty:
        raise ValueError("Cannot run inference on an empty DataFrame.")

    if feature_schema is not None:
        X = feature_schema.validate(flow_data)
    else:
        norm_cols = [str(c).strip() for c in flow_data.columns]
        seen: Dict[str, int] = {}
        for c in norm_cols:
            seen[c] = seen.get(c, 0) + 1
        dups = [col for col, count in seen.items() if count > 1]
        if dups:
            raise ValueError(f"Duplicate column names detected after normalization: {dups}")
        X = flow_data.copy(deep=False)
        X.columns = norm_cols
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [f for f in expected if f not in set(X.columns)]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            X = X[expected]

    predictions = model.predict(X)

    # Validate probability support and resolve positive class index dynamically
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"Estimator '{type(model).__name__}' does not support probability estimation (predict_proba missing)."
        )

    probs = model.predict_proba(X)
    classes = getattr(model, "classes_", None)

    if classes is None or pos_label not in classes:
        raise ValueError(
            f"Positive class '{pos_label}' not found in estimator classes: {list(classes) if classes is not None else None}."
        )

    pos_idx = list(classes).index(pos_label)
    probabilities = probs[:, pos_idx]

    # Verify probability bounds and finiteness
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Estimator produced non-finite (NaN or Inf) probability scores.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Estimator produced probabilities outside the valid range [0.0, 1.0].")

    return predictions, probabilities
