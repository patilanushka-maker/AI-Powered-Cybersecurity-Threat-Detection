"""
Alert Engine Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Evaluating prediction probabilities and applying risk thresholds
- Assigning alert severity tiers (Low, Medium, High, Critical)
- Generating structured security events from batch inference predictions
- Exporting alerts to JSON/JSONL without fabricating network evidence
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


def categorize_alert_severity(
    probability: Optional[float], attack_type: str = "Unknown"
) -> str:
    """
    Determine alert severity based on confidence probability and attack taxonomy.

    Severity Policy & Probability Triage Thresholds:
    - [0.00, 0.50): BENIGN (normal baseline traffic, not flagged as attack)
    - [0.50, 0.70): LOW (low confidence anomaly signal)
    - [0.70, 0.85): MEDIUM (moderate confidence threat indicator)
    - [0.85, 0.95): HIGH (high confidence attack signature)
    - [0.95, 1.00]: CRITICAL (very high confidence attack signature)

    Handling of Missing / Non-Finite Scores:
    - If probability is None or non-finite (NaN, Inf), returns 'UNKNOWN'.

    Parameters
    ----------
    probability : Optional[float]
        Estimated intrusion probability score in [0.0, 1.0].
    attack_type : str, default "Unknown"
        Identified attack family (e.g., DoS, Infiltration, Brute Force).

    Returns
    -------
    str
        Severity label: 'BENIGN', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', or 'UNKNOWN'.

    Raises
    ------
    ValueError
        If probability is finite but outside the valid range [0.0, 1.0].
    """
    if probability is None or not np.isfinite(probability):
        return "UNKNOWN"

    p = float(probability)
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"Probability score must be in range [0.0, 1.0], got {p}."
        )

    if p < 0.5:
        return "BENIGN"
    elif p < 0.7:
        return "LOW"
    elif p < 0.85:
        return "MEDIUM"
    elif p < 0.95:
        return "HIGH"
    return "CRITICAL"


def generate_security_alert(
    flow_id: str,
    attack_type: str,
    probability: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format a structured security alert event.

    Parameters
    ----------
    flow_id : str
        Unique flow or record identifier.
    attack_type : str
        Identified attack type.
    probability : Optional[float]
        Model confidence probability.
    metadata : Optional[Dict[str, Any]], default None
        Genuine flow attributes present in telemetry (IP, port, protocol, etc.).

    Returns
    -------
    Dict[str, Any]
        Structured alert payload.
    """
    severity = categorize_alert_severity(probability, attack_type)
    prob_val = (
        round(float(probability), 4)
        if (probability is not None and np.isfinite(probability))
        else None
    )
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "flow_id": str(flow_id),
        "attack_type": str(attack_type),
        "probability": prob_val,
        "severity": severity,
        "is_model_prediction": True,
        "triage_disclaimer": "Prototype triage prioritization signal; not proof of confirmed incident.",
        "metadata": metadata or {},
    }


def process_batch_alerts(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    flow_data: pd.DataFrame,
    threshold: float = 0.5,
    id_column: Optional[str] = None,
    attack_type: str = "Predicted Attack",
) -> List[Dict[str, Any]]:
    """
    Generate structured alerts for predicted attacks from a batch inference run.

    Only rows predicted as an attack (prediction == 1 or probability >= threshold)
    produce alerts. Benign predictions never generate attack alerts.

    Parameters
    ----------
    predictions : np.ndarray
        Array of binary predictions (0 = Benign, 1 = Attack).
    probabilities : np.ndarray
        Array of attack probabilities.
    flow_data : pd.DataFrame
        Input DataFrame containing source flow records.
    threshold : float, default 0.5
        Threshold above which a flow is considered anomalous.
    id_column : Optional[str], default None
        Optional column name in flow_data containing flow IDs.
    attack_type : str, default 'Predicted Attack'
        Label describing the attack type.

    Returns
    -------
    List[Dict[str, Any]]
        List of generated alert event dictionaries.
    """
    if len(predictions) != len(probabilities) or len(predictions) != len(flow_data):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}), "
            f"probabilities ({len(probabilities)}), flow_data ({len(flow_data)})."
        )

    alerts: List[Dict[str, Any]] = []

    # Known candidate metadata columns to attach if present in source data
    candidate_meta_cols = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Source Port",
        "Destination Port",
        "Protocol",
    ]

    for i in range(len(predictions)):
        pred = int(predictions[i])
        prob = float(probabilities[i])

        # Filter: Only generate alerts for attack classifications
        if pred == 1 or prob >= threshold:
            # Determine flow ID safely
            if id_column is not None and id_column in flow_data.columns:
                flow_id = str(flow_data[id_column].iloc[i])
            elif "Flow ID" in flow_data.columns:
                flow_id = str(flow_data["Flow ID"].iloc[i])
            else:
                flow_id = f"flow_index_{i}"

            # Extract genuinely available metadata without fabricating evidence
            row_metadata: Dict[str, Any] = {}
            for col in candidate_meta_cols:
                if col in flow_data.columns:
                    val = flow_data[col].iloc[i]
                    if pd.notna(val):
                        if isinstance(val, (int, np.integer)):
                            row_metadata[col] = int(val)
                        elif isinstance(val, (float, np.floating)):
                            row_metadata[col] = float(val)
                        else:
                            row_metadata[col] = str(val)

            alert = generate_security_alert(
                flow_id=flow_id,
                attack_type=attack_type,
                probability=prob,
                metadata=row_metadata,
            )
            alerts.append(alert)

    return alerts


def save_alerts_json(
    alerts: List[Dict[str, Any]],
    output_path: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """
    Save generated alerts to a formatted JSON file.

    Parameters
    ----------
    alerts : List[Dict[str, Any]]
        List of alert dictionaries.
    output_path : Union[str, Path]
        Destination JSON file path.
    overwrite : bool, default False
        Whether to overwrite existing file.

    Returns
    -------
    Path
        Path to saved file.

    Raises
    ------
    FileExistsError
        If output_path exists and overwrite=False.
    """
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Alerts output file already exists at '{path}'. Pass overwrite=True to overwrite."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    return path
