"""
Unit and integration tests for Feature Schema validation and whitespace normalization fix.

Tests:
- Valid CSV with expected feature schema
- Column names containing surrounding whitespace (leading and trailing)
- Genuinely missing required features with clear error messages
- Duplicate column names detected after whitespace normalization
- Correct feature ordering preserved matching model expectation
- Prediction output shape and valid binary labels
- Real CIC-IDS2017 raw CSV slice end-to-end inference via dashboard batch predictor
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import FeatureSchema
from src.predict import predict_network_flows
from src.train import train_model
from app import load_trusted_model, run_batch_prediction_on_dataframe


def test_feature_schema_with_surrounding_whitespace():
    """
    Verify that FeatureSchema.validate strips leading and trailing whitespace
    from column names and succeeds when columns match the schema.
    """
    schema = FeatureSchema(
        feature_names=["Destination Port", "Flow Duration", "Total Fwd Packets"],
        numeric_features=["Destination Port", "Flow Duration", "Total Fwd Packets"],
    )

    # Input DataFrame with messy surrounding whitespace
    df = pd.DataFrame({
        " Destination Port": [80, 443],
        " Flow Duration ": [12000, 34000],
        "Total Fwd Packets ": [5, 12],
    })

    validated = schema.validate(df)
    assert list(validated.columns) == ["Destination Port", "Flow Duration", "Total Fwd Packets"]
    assert len(validated) == 2
    assert validated["Destination Port"].iloc[0] == 80


def test_feature_schema_duplicate_columns_after_normalization():
    """
    Verify that if two columns in the input DataFrame resolve to the same name
    after stripping whitespace, a clear ValueError is raised.
    """
    schema = FeatureSchema(
        feature_names=["Destination Port", "Flow Duration"],
        numeric_features=["Destination Port", "Flow Duration"],
    )

    # Two columns that collide after whitespace normalization
    df = pd.DataFrame({
        " Destination Port": [80, 443],
        "Destination Port": [8080, 8443],
        "Flow Duration": [1000, 2000],
    })

    with pytest.raises(ValueError, match="Duplicate column names detected after normalization"):
        schema.validate(df)


def test_feature_schema_missing_required_features():
    """
    Verify that if genuinely required features are missing, FeatureSchema.validate
    raises a clear error identifying the exact missing features.
    """
    schema = FeatureSchema(
        feature_names=["Destination Port", "Flow Duration", "ACK Flag Count"],
        numeric_features=["Destination Port", "Flow Duration", "ACK Flag Count"],
    )

    df = pd.DataFrame({
        " Destination Port": [80, 443],
        "Extra Column": [1, 2],
    })

    with pytest.raises(ValueError, match="Input DataFrame is missing required features") as exc_info:
        schema.validate(df)

    err_msg = str(exc_info.value)
    assert "Flow Duration" in err_msg
    assert "ACK Flag Count" in err_msg


def test_feature_schema_preserves_feature_ordering():
    """
    Verify that FeatureSchema.validate returns columns in the exact order
    defined by feature_names, regardless of input column order.
    """
    schema = FeatureSchema(
        feature_names=["feat_a", "feat_b", "feat_c"],
        numeric_features=["feat_a", "feat_b", "feat_c"],
    )

    # Scrambled input order with whitespace
    df = pd.DataFrame({
        " feat_c ": [3.0],
        "feat_a": [1.0],
        " feat_b": [2.0],
    })

    validated = schema.validate(df)
    assert list(validated.columns) == ["feat_a", "feat_b", "feat_c"]
    assert validated.iloc[0].tolist() == [1.0, 2.0, 3.0]


def test_predict_network_flows_with_whitespace_and_schema():
    """
    Verify predict_network_flows produces correct output shapes and binary labels
    when fed inputs with column whitespace.
    """
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    y_train = pd.Series([0, 0, 1, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)
    schema = FeatureSchema(feature_names=["f1", "f2"], numeric_features=["f1", "f2"])

    # Test input with whitespace
    test_df = pd.DataFrame({
        " f1 ": [1.5, 3.5],
        " f2": [15.0, 35.0],
    })

    preds, probs = predict_network_flows(pipe, test_df, feature_schema=schema)
    assert preds.shape == (2,)
    assert probs.shape == (2,)
    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_predict_network_flows_without_schema_normalizes_and_orders():
    """
    Verify predict_network_flows normalizes columns and respects model.feature_names_in_
    even when feature_schema is None.
    """
    X_train = pd.DataFrame({"alpha": [1.0, 2.0], "beta": [10.0, 20.0]})
    y_train = pd.Series([0, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)

    # Input with scrambled order and leading whitespace
    test_df = pd.DataFrame({
        " beta ": [15.0, 25.0],
        " alpha": [1.2, 2.2],
    })

    preds, probs = predict_network_flows(pipe, test_df, feature_schema=None)
    assert len(preds) == 2
    assert len(probs) == 2


def test_batch_prediction_dashboard_on_whitespace_dataframe():
    """
    Verify run_batch_prediction_on_dataframe in app.py handles raw DataFrames
    with leading/trailing whitespace without raising Feature Schema Error.
    """
    X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y_train = pd.Series([0, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)

    raw_df = pd.DataFrame({
        " f1": [1.5, 2.5],
        "f2 ": [3.5, 4.5],
        "Label": ["BENIGN", "ATTACK"],
    })

    pred_df, alerts, err = run_batch_prediction_on_dataframe(pipe, meta, raw_df)
    assert err is None
    assert pred_df is not None
    assert len(pred_df) == 2
    assert "predicted_class" in pred_df.columns
    assert "predicted_label" in pred_df.columns
    assert "attack_score" in pred_df.columns
    assert list(pred_df["predicted_label"]) in [["Benign", "Attack"], ["Benign", "Benign"], ["Attack", "Attack"]]


def test_batch_prediction_dashboard_handles_duplicate_error():
    """
    Verify run_batch_prediction_on_dataframe returns a clear error if input
    contains duplicate column names after normalization.
    """
    X_train = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    y_train = pd.Series([0, 1])
    pipe, meta = train_model(X_train, y_train, random_state=42)

    raw_df = pd.DataFrame({
        " f1": [1.5, 2.5],
        "f1 ": [3.5, 4.5],
        "f2": [5.0, 6.0],
    })

    pred_df, alerts, err = run_batch_prediction_on_dataframe(pipe, meta, raw_df)
    assert pred_df is None
    assert err is not None
    assert "Duplicate column names detected after normalization" in err


def test_cicids2017_actual_raw_dataset_prediction_end_to_end():
    """
    Integration test: Load the active default model from models/ and execute
    run_batch_prediction_on_dataframe on rows loaded directly from the raw
    Friday DDoS CSV containing the authentic 65 leading-whitespace columns.
    """
    model, metadata, load_err = load_trusted_model()
    assert load_err is None
    assert model is not None
    assert metadata["n_features"] == 78

    csv_path = Path("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    assert csv_path.exists(), "Raw Friday DDoS dataset must exist for verification."

    # Read using standard pd.read_csv (simulating Streamlit st.file_uploader)
    raw_df = pd.read_csv(csv_path, nrows=50)
    assert any(c.startswith(" ") for c in raw_df.columns), "Raw CSV must contain leading spaces."

    pred_df, alerts, err = run_batch_prediction_on_dataframe(
        model=model,
        metadata=metadata,
        raw_df=raw_df,
        threshold=0.50,
    )

    assert err is None, f"Expected no error, got: {err}"
    assert pred_df is not None
    assert len(pred_df) == 50
    assert "predicted_class" in pred_df.columns
    assert "predicted_label" in pred_df.columns
    assert "attack_score" in pred_df.columns
    assert set(pred_df["predicted_class"]).issubset({0, 1})
    assert np.all((pred_df["attack_score"] >= 0.0) & (pred_df["attack_score"] <= 1.0))
