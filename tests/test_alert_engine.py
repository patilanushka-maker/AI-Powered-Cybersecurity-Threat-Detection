"""
Regression and unit tests for the security alert engine.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.alert_engine import (
    categorize_alert_severity,
    generate_security_alert,
    process_batch_alerts,
    save_alerts_json,
)


def test_categorize_alert_severity_thresholds():
    """Verify severity tier mapping across probability thresholds."""
    assert categorize_alert_severity(0.0) == "BENIGN"
    assert categorize_alert_severity(0.2) == "BENIGN"
    assert categorize_alert_severity(0.5) == "LOW"
    assert categorize_alert_severity(0.6) == "LOW"
    assert categorize_alert_severity(0.7) == "MEDIUM"
    assert categorize_alert_severity(0.8) == "MEDIUM"
    assert categorize_alert_severity(0.85) == "HIGH"
    assert categorize_alert_severity(0.9) == "HIGH"
    assert categorize_alert_severity(0.95) == "CRITICAL"
    assert categorize_alert_severity(0.99) == "CRITICAL"
    assert categorize_alert_severity(1.0) == "CRITICAL"


def test_generate_security_alert_structure():
    """Verify that generated security alerts adhere to the required schema."""
    alert = generate_security_alert(
        flow_id="FLOW_1001",
        attack_type="DoS Hulk",
        probability=0.92,
        metadata={"src_ip": "192.168.1.50", "dst_port": 80},
    )

    assert alert["flow_id"] == "FLOW_1001"
    assert alert["attack_type"] == "DoS Hulk"
    assert alert["probability"] == 0.92
    assert alert["severity"] == "HIGH"
    assert "timestamp" in alert
    assert alert["metadata"]["src_ip"] == "192.168.1.50"
    assert alert["is_model_prediction"] is True


def test_generate_security_alert_default_metadata():
    """Regression test: alert engine handles None metadata gracefully."""
    alert = generate_security_alert(
        flow_id="FLOW_1002",
        attack_type="PortScan",
        probability=0.45,
        metadata=None,
    )
    assert alert["severity"] == "BENIGN"
    assert alert["metadata"] == {}


def test_process_batch_alerts_attack_and_benign_filtering():
    """Verify that only attack predictions produce alerts; benign flows are filtered."""
    df = pd.DataFrame(
        {
            "Flow ID": ["F_001", "F_002", "F_003"],
            "Source IP": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
            "Destination Port": [80, 443, 8080],
        }
    )
    # Row 0: Benign (pred=0, prob=0.1) -> No alert
    # Row 1: Attack (pred=1, prob=0.88) -> Alert
    # Row 2: Benign (pred=0, prob=0.3) -> No alert
    preds = np.array([0, 1, 0])
    probs = np.array([0.10, 0.88, 0.30])

    alerts = process_batch_alerts(preds, probs, df, threshold=0.5)

    assert len(alerts) == 1
    assert alerts[0]["flow_id"] == "F_002"
    assert alerts[0]["severity"] == "HIGH"
    assert alerts[0]["metadata"]["Source IP"] == "192.168.1.2"
    assert alerts[0]["metadata"]["Destination Port"] == 443


def test_process_batch_alerts_zero_attacks():
    """Verify that zero attack predictions produce an empty list without error."""
    df = pd.DataFrame({"feat": [1, 2]})
    preds = np.array([0, 0])
    probs = np.array([0.05, 0.12])

    alerts = process_batch_alerts(preds, probs, df, threshold=0.5)
    assert alerts == []


def test_process_batch_alerts_length_mismatch_raises():
    """Verify ValueError on mismatched input lengths."""
    df = pd.DataFrame({"feat": [1, 2]})
    with pytest.raises(ValueError) as exc:
        process_batch_alerts(np.array([1]), np.array([0.8]), df)
    assert "Length mismatch" in str(exc.value)


def test_save_alerts_json_and_overwrite_protection(tmp_path: Path):
    """Verify JSON alert serialization and overwrite protection."""
    alerts = [{"flow_id": "test_1", "severity": "HIGH", "probability": 0.9}]
    out_file = tmp_path / "alerts.json"

    saved = save_alerts_json(alerts, out_file, overwrite=False)
    assert saved.exists()

    with open(saved) as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["flow_id"] == "test_1"

    # Fails without overwrite=True
    with pytest.raises(FileExistsError):
        save_alerts_json(alerts, out_file, overwrite=False)

    # Succeeds with overwrite=True
    saved_again = save_alerts_json(alerts, out_file, overwrite=True)
    assert saved_again.exists()


def test_categorize_alert_severity_boundaries_and_invalid():
    """Verify exact boundary values and out-of-range handling."""
    # Exact boundaries
    assert categorize_alert_severity(0.0) == "BENIGN"
    assert categorize_alert_severity(0.49999) == "BENIGN"
    assert categorize_alert_severity(0.5) == "LOW"
    assert categorize_alert_severity(0.69999) == "LOW"
    assert categorize_alert_severity(0.7) == "MEDIUM"
    assert categorize_alert_severity(0.84999) == "MEDIUM"
    assert categorize_alert_severity(0.85) == "HIGH"
    assert categorize_alert_severity(0.94999) == "HIGH"
    assert categorize_alert_severity(0.95) == "CRITICAL"
    assert categorize_alert_severity(1.0) == "CRITICAL"

    # Missing / NaN / Inf scores
    assert categorize_alert_severity(None) == "UNKNOWN"
    assert categorize_alert_severity(np.nan) == "UNKNOWN"
    assert categorize_alert_severity(float("nan")) == "UNKNOWN"
    assert categorize_alert_severity(np.inf) == "UNKNOWN"

    # Out of range finite scores raise ValueError
    with pytest.raises(ValueError):
        categorize_alert_severity(-0.01)
    with pytest.raises(ValueError):
        categorize_alert_severity(1.01)


def test_generate_security_alert_triage_disclaimer():
    """Verify that alerts contain triage disclaimer and handle None probability."""
    alert = generate_security_alert("F1", "DDoS", None)
    assert alert["severity"] == "UNKNOWN"
    assert alert["probability"] is None
    assert "triage_disclaimer" in alert
    assert "triage" in alert["triage_disclaimer"].lower()


