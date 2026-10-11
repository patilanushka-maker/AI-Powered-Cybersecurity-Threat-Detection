"""
Automated unit tests for main.py CLI workflows.
"""

from pathlib import Path
import json
import pandas as pd
import pytest
from main import build_arg_parser, main


@pytest.fixture
def synthetic_labeled_csv(tmp_path: Path) -> Path:
    """Create a temporary synthetic labeled CSV."""
    csv_path = tmp_path / "synthetic_network_flows.csv"
    data = {
        "Source IP": ["192.168.1.1"] * 5 + ["10.0.0.1"] * 5,
        "Destination Port": [80, 443, 80, 22, 443, 8080, 80, 8080, 21, 80],
        "Flow Duration": [100.0, 150.0, 120.0, 90.0, 110.0, 800.0, 950.0, 850.0, 900.0, 880.0],
        "Packet Count": [5, 6, 4, 3, 5, 50, 65, 55, 60, 58],
        "Label": ["BENIGN"] * 5 + ["DDoS"] * 5,
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_unlabeled_csv(tmp_path: Path) -> Path:
    """Create a temporary synthetic unlabeled flow CSV for prediction."""
    csv_path = tmp_path / "unlabeled_flows.csv"
    data = {
        "Source IP": ["192.168.1.5", "10.0.0.99"],
        "Destination Port": [80, 8080],
        "Flow Duration": [115.0, 920.0],
        "Packet Count": [5, 62],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_cli_help():
    """Verify CLI argument parser provides --help without errors."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_train_and_evaluate_workflow(tmp_path: Path, synthetic_labeled_csv: Path):
    """Test full train and evaluate workflow via CLI using temporary output directories."""
    model_path = tmp_path / "model.joblib"
    train_report = tmp_path / "train_report.json"
    eval_report = tmp_path / "eval_report.json"

    # 1. Run CLI train
    train_args = [
        "--mode", "train",
        "--data-path", str(synthetic_labeled_csv),
        "--model-output", str(model_path),
        "--report-output", str(train_report),
        "--test-size", "0.3",
        "--random-state", "42",
    ]
    exit_code = main(train_args)
    assert exit_code == 0
    assert model_path.exists()
    assert train_report.exists()

    with open(train_report) as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "false_positive_rate" in metrics

    # 2. Run CLI evaluate
    eval_args = [
        "--mode", "evaluate",
        "--data-path", str(synthetic_labeled_csv),
        "--model-path", str(model_path),
        "--report-output", str(eval_report),
    ]
    exit_code_eval = main(eval_args)
    assert exit_code_eval == 0
    assert eval_report.exists()


def test_cli_predict_and_alerts_workflow(
    tmp_path: Path, synthetic_labeled_csv: Path, synthetic_unlabeled_csv: Path
):
    """Test full prediction and threat alerts generation via CLI."""
    model_path = tmp_path / "model.joblib"
    pred_path = tmp_path / "predictions.csv"
    alerts_path = tmp_path / "alerts.json"

    # Train a model first
    train_args = [
        "--mode", "train",
        "--data-path", str(synthetic_labeled_csv),
        "--model-output", str(model_path),
        "--report-output", str(tmp_path / "tr_rep.json"),
    ]
    assert main(train_args) == 0

    # Run CLI predict
    predict_args = [
        "--mode", "predict",
        "--model-path", str(model_path),
        "--data-path", str(synthetic_unlabeled_csv),
        "--predictions-output", str(pred_path),
        "--alerts-output", str(alerts_path),
        "--threshold", "0.5",
    ]
    exit_code = main(predict_args)
    assert exit_code == 0
    assert pred_path.exists()
    assert alerts_path.exists()

    pred_df = pd.read_csv(pred_path)
    assert len(pred_df) == 2
    assert "predicted_class" in pred_df.columns
    assert "attack_score" in pred_df.columns

    with open(alerts_path) as f:
        alerts = json.load(f)
    # The second flow (duration 920, packets 62) resembles attack flows
    assert isinstance(alerts, list)
    for alert in alerts:
        assert "flow_id" in alert
        assert "severity" in alert
        assert "probability" in alert
        assert alert["is_model_prediction"] is True


def test_cli_overwrite_protection(tmp_path: Path, synthetic_labeled_csv: Path):
    """Verify CLI enforces overwrite protection on existing artifacts."""
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    args = [
        "--mode", "train",
        "--data-path", str(synthetic_labeled_csv),
        "--model-output", str(model_path),
        "--report-output", str(report_path),
    ]
    # First execution succeeds
    assert main(args) == 0

    # Second execution without --overwrite fails with exit code 1
    assert main(args) == 1

    # Second execution with --overwrite succeeds
    assert main(args + ["--overwrite"]) == 0


def test_cli_negative_missing_model_file(tmp_path: Path, synthetic_unlabeled_csv: Path):
    """Verify clear error when model artifact does not exist."""
    args = [
        "--mode", "predict",
        "--model-path", str(tmp_path / "non_existent.joblib"),
        "--data-path", str(synthetic_unlabeled_csv),
    ]
    assert main(args) == 1


def test_cli_training_hyperparameters_and_sampling(tmp_path: Path, synthetic_labeled_csv: Path):
    """Verify CLI handles --max-samples, --n-estimators, and --max-depth."""
    model_path = tmp_path / "sampled_model.joblib"
    report_path = tmp_path / "sampled_report.json"

    args = [
        "--mode", "train",
        "--data-path", str(synthetic_labeled_csv),
        "--model-output", str(model_path),
        "--report-output", str(report_path),
        "--max-samples", "6",
        "--n-estimators", "10",
        "--max-depth", "5",
        "--test-size", "0.33",
        "--overwrite",
    ]
    assert main(args) == 0
    assert model_path.exists()
    assert report_path.exists()

    with open(report_path) as f:
        rep = json.load(f)
    assert rep["sampling_info"]["sampling_applied"] is True
    assert rep["sampling_info"]["rows_retained"] == 6
    assert rep["model_configuration"]["n_estimators"] == 10
    assert rep["model_configuration"]["max_depth"] == 5


def test_cli_train_with_data_dir_directory(tmp_path: Path):
    """Verify CLI train mode functions correctly when --data-dir points to a directory of CSVs."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    (data_dir / "friday.csv").write_text("Flow Duration,Tot Fwd Pkts,Label\n100,5,BENIGN\n200,10,DDoS\n")
    (data_dir / "tuesday.csv").write_text("Flow Duration,Tot Fwd Pkts,Label\n300,15,BENIGN\n400,20,FTP-Patator\n")

    model_path = tmp_path / "multi_model.joblib"
    report_path = tmp_path / "multi_report.json"

    args = [
        "train",
        "--data-dir", str(data_dir),
        "--model-output", str(model_path),
        "--report-output", str(report_path),
        "--test-size", "0.5",
        "--overwrite",
    ]
    assert main(args) == 0
    assert model_path.exists()
    assert report_path.exists()
    with open(report_path) as f:
        rep = json.load(f)
    assert rep["total_rows_loaded"] == 4


def test_cli_experiment_subcommand_missing_file(tmp_path: Path):
    """Verify CLI experiment mode handles missing file paths gracefully."""
    args = [
        "experiment",
        "--train-paths", str(tmp_path / "nonexistent1.csv"),
        "--independent-test-path", str(tmp_path / "nonexistent2.csv"),
    ]
    exit_code = main(args)
    assert exit_code == 1
