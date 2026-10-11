"""
CLI Orchestrator for AI-Powered Cybersecurity Threat Detection System.

Supports three operational modes:
1. train: Ingests labeled flow data, executes leakage-aware splitting, trains Random Forest pipeline,
   evaluates on held-out test split, and persists model artifacts and JSON reports.
2. evaluate: Loads a trained model artifact and evaluates performance on a labeled validation/test CSV.
3. predict: Loads a trained model artifact, runs batch scoring on network flows, exports row-level predictions,
   and generates structured threat alerts for detected attacks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import (
    handle_non_finite_values,
    encode_binary_target,
    train_test_split_network_flows,
    split_data_temporal,
    sample_dataset,
    DEFAULT_BENIGN_LABELS,
    DEFAULT_RECOGNIZED_ATTACK_LABELS,
)
from src.feature_engineering import prepare_feature_matrix, FeatureSchema
from src.train import (
    train_model,
    save_model_artifact,
    DEFAULT_MODEL_PATH,
)
from src.evaluate import evaluate_model, save_evaluation_report
from src.predict import load_model_for_inference, predict_network_flows
from src.alert_engine import process_batch_alerts, save_alerts_json
from src.dataset_validator import validate_dataset_directory
from src.experiments import run_milestone16_experiments
from src.utils import setup_logger


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="threat-detection-cli",
        description="AI-Powered Cybersecurity Threat Detection CLI Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Allow either positional command or --mode flag
    parser.add_argument(
        "subcommand",
        nargs="?",
        default=None,
        choices=["train", "evaluate", "predict", "validate-data", "experiment"],
        help="Operational command (positional alias for --mode).",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "validate-data", "experiment"],
        default=None,
        help="Operational mode to execute.",
    )

    # Data options
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to input network flow CSV dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw network flow CSV files for validate-data mode.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Label",
        help="Name of target label column in labeled datasets.",
    )
    parser.add_argument(
        "--train-paths",
        type=str,
        default="data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,data/raw/Tuesday-WorkingHours.pcap_ISCX.csv",
        help="Comma-separated paths to development dataset captures for experiment mode.",
    )
    parser.add_argument(
        "--independent-test-path",
        type=str,
        default="data/raw/Wednesday-workingHours.pcap_ISCX.csv",
        help="Path to external independent test dataset capture for experiment mode.",
    )

    # Model artifact options
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to serialized model artifact (.joblib) for evaluate/predict/validate-data modes.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output destination path for trained model artifact (.joblib) in train mode.",
    )

    # Output options
    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Destination path for evaluation report JSON file.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Alias for --report-output (used in validate-data mode).",
    )
    parser.add_argument(
        "--predictions-output",
        type=str,
        default="outputs/predictions/predictions.csv",
        help="Destination path for batch scoring output CSV in predict mode.",
    )
    parser.add_argument(
        "--alerts-output",
        type=str,
        default="outputs/predictions/alerts.json",
        help="Destination path for structured security alerts JSON in predict mode.",
    )
    parser.add_argument(
        "--cm-plot-output",
        type=str,
        default=None,
        help="Optional path to export confusion matrix image plot (e.g. outputs/reports/cm.png).",
    )

    # Validation options
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows per processing chunk for memory-aware dataset validation.",
    )
    parser.add_argument(
        "--duplicate-check",
        choices=["within_file", "cross_file", "none"],
        default="within_file",
        help="Duplicate flow detection strategy: 'within_file', 'cross_file', or 'none'.",
    )

    # Training and splitting hyperparameters
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data allocated to held-out test split in train mode.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Reproducibility seed for train/test split and model estimators.",
    )
    parser.add_argument(
        "--split-method",
        choices=["stratified", "temporal"],
        default="stratified",
        help="Data splitting method for train mode.",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="Timestamp",
        help="Timestamp column name required when --split-method is temporal.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum row limit for reproducible, stratified subsampling for memory safety.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of decision trees for Random Forest classifier.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional maximum tree depth for Random Forest classifier to bound memory usage.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying attack and generating alerts in predict mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing model, report, and prediction files.",
    )

    return parser


def run_train(args: argparse.Namespace, logger) -> int:
    """Execute training workflow."""
    logger.info("Executing training workflow...")
    data_path = Path(args.data_path or args.data_dir)
    model_out = Path(args.model_output)
    report_out = Path(args.report_out or args.report_output or "outputs/reports/train_evaluation.json")

    if args.max_samples is not None and args.max_samples <= 0:
        logger.error("--max-samples must be a positive integer (> 0), got %d", args.max_samples)
        return 1

    # Pre-check overwrite rules before expensive computation
    if model_out.exists() and not args.overwrite:
        logger.error(
            "Model output path '%s' already exists. Use --overwrite to replace.", model_out
        )
        return 1

    if report_out.exists() and not args.overwrite:
        logger.error(
            "Report output path '%s' already exists. Use --overwrite to replace.", report_out
        )
        return 1

    logger.info("Loading raw dataset from %s", data_path)
    df = load_dataset(data_path)
    logger.info("Ingested %d rows and %d columns", len(df), len(df.columns))

    df_clean = handle_non_finite_values(df)

    # Optional controlled sampling for resource-constrained environments
    df_trainable, sampling_info = sample_dataset(
        df_clean,
        max_samples=args.max_samples,
        target_column=args.target_column,
        random_state=args.random_state,
    )
    if sampling_info["sampling_applied"]:
        logger.info(
            "Controlled sampling applied (%s): %d rows retained out of %d (%.2f%%)",
            sampling_info["sampling_method"],
            sampling_info["rows_retained"],
            sampling_info["total_rows_available"],
            sampling_info["sampling_ratio"] * 100.0,
        )

    if args.split_method == "temporal":
        logger.info("Performing chronological temporal split using column '%s'...", args.time_column)
        split_ratio = 1.0 - args.test_size
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_data_temporal(
            df_trainable,
            time_column=args.time_column,
            split_ratio=split_ratio,
            target_column=args.target_column,
        )
        # Prepare features on train and test
        X_train, _, schema = prepare_feature_matrix(
            X_train_raw, target_column=None, drop_metadata=True
        )
        X_test = schema.validate(X_test_raw)
        y_train = encode_binary_target(y_train_raw)
        y_test = encode_binary_target(y_test_raw)
    else:
        logger.info("Preparing feature matrix and encoding labels...")
        X, y_raw, schema = prepare_feature_matrix(
            df_trainable, target_column=args.target_column, drop_metadata=True
        )
        y = encode_binary_target(y_raw)

        logger.info("Executing stratified train/test split (test_size=%.2f)...", args.test_size)
        X_train, X_test, y_train, y_test = train_test_split_network_flows(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=True
        )

    logger.info("Training features: %d samples, %d features", len(X_train), len(X_train.columns))
    logger.info("Testing features: %d samples", len(X_test))

    pipeline, metadata = train_model(
        X_train=X_train,
        y_train=y_train,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        feature_schema=schema,
    )
    logger.info("Training complete. Evaluating on held-out test split...")

    eval_results = evaluate_model(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        pos_label=1,
        save_confusion_matrix_plot=args.cm_plot_output,
    )

    # Document comprehensive evaluation context and limitations
    eval_results["evaluation_type"] = "held_out_partition"
    eval_results["dataset_source"] = str(data_path)
    eval_results["total_rows_loaded"] = len(df)
    eval_results["train_samples_count"] = len(X_train)
    eval_results["test_samples_count"] = len(X_test)
    eval_results["feature_count"] = len(schema.feature_names)
    eval_results["feature_names"] = schema.feature_names
    eval_results["metadata_excluded"] = schema.metadata_excluded
    eval_results["split_strategy"] = args.split_method
    eval_results["test_size"] = args.test_size
    eval_results["random_seed"] = args.random_state
    eval_results["train_class_distribution"] = {str(k): int(v) for k, v in y_train.value_counts().items()}
    eval_results["test_class_distribution"] = {str(k): int(v) for k, v in y_test.value_counts().items()}
    eval_results["target_label_policy"] = {
        "benign_labels": sorted(list(DEFAULT_BENIGN_LABELS)),
        "recognized_attack_labels": sorted(list(DEFAULT_RECOGNIZED_ATTACK_LABELS)),
        "unknown_action": "reject_with_error",
        "missing_or_blank_action": "reject_with_error",
        "binary_mapping": "0: Benign, 1: Attack",
        "multiclass_claimed": False,
    }
    eval_results["model_configuration"] = metadata.get("estimator_params", {})
    eval_results["evaluation_method"] = "Scikit-Learn Binary Metrics on Held-Out Split"
    eval_results["sampling_info"] = sampling_info
    eval_results["known_leakage_limitations"] = [
        "Temporal splitting preserves chronological order but does NOT eliminate duplicate-flow, host, or session-level leakage across split boundaries.",
        "Stratified random splitting preserves class ratios but risks flow record and host session leakage between train and test partitions.",
        "Identical 5-tuple flow records appearing in both train and test partitions inflate cross-validation metrics.",
        "Binary classification aggregates all attack classes into class 1; it does not claim to distinguish individual attack categories.",
    ]
    eval_results["data_overlap_note"] = (
        "Metrics reflect performance on the held-out split of the provided training dataset. "
        "No claim of cross-dataset generalization is made without external independent validation."
    )

    logger.info("Evaluation Results (Held-Out Test Partition):")
    logger.info("  Accuracy:            %.4f", eval_results["accuracy"])
    logger.info("  Precision:           %.4f", eval_results["precision"])
    logger.info("  Recall:              %.4f", eval_results["recall"])
    logger.info("  F1-Score:            %.4f", eval_results["f1_score"])
    logger.info("  False Positive Rate: %.4f", eval_results["false_positive_rate"])

    save_model_artifact(pipeline, metadata, model_out, overwrite=args.overwrite)
    logger.info("Saved trained model artifact to %s", model_out)

    save_evaluation_report(eval_results, report_out, overwrite=args.overwrite)
    logger.info("Saved evaluation report to %s", report_out)

    return 0


def run_evaluate(args: argparse.Namespace, logger) -> int:
    """Execute evaluation workflow."""
    logger.info("Executing evaluation workflow...")
    model_path = Path(args.model_path)
    data_path = Path(args.data_path or args.data_dir)
    report_out = Path(args.report_out or args.report_output or "outputs/reports/evaluation_report.json")

    if not model_path.exists():
        logger.error("Model artifact not found at '%s'. Run in train mode first.", model_path)
        return 1

    if report_out.exists() and not args.overwrite:
        logger.error(
            "Report output path '%s' already exists. Use --overwrite to replace.", report_out
        )
        return 1

    logger.info("Loading model artifact from %s", model_path)
    pipeline, metadata = load_model_for_inference(model_path)

    schema = None
    if metadata.get("feature_schema"):
        schema = FeatureSchema.from_dict(metadata["feature_schema"])

    logger.info("Loading evaluation dataset from %s", data_path)
    df = load_dataset(data_path)
    df_clean = handle_non_finite_values(df)

    X_raw, y_raw, _ = prepare_feature_matrix(
        df_clean, target_column=args.target_column, drop_metadata=True
    )
    y = encode_binary_target(y_raw)

    if schema is not None:
        X = schema.validate(X_raw)
    else:
        X = X_raw

    logger.info("Evaluating model on %d samples...", len(X))
    eval_results = evaluate_model(
        model=pipeline,
        X_test=X,
        y_test=y,
        pos_label=1,
        save_confusion_matrix_plot=args.cm_plot_output,
    )

    eval_results["evaluation_type"] = "external_dataset"
    eval_results["dataset_source"] = str(data_path)
    eval_results["model_path"] = str(model_path)
    eval_results["total_samples_evaluated"] = len(X)
    eval_results["target_label_policy"] = {
        "benign_labels": sorted(list(DEFAULT_BENIGN_LABELS)),
        "recognized_attack_labels": sorted(list(DEFAULT_RECOGNIZED_ATTACK_LABELS)),
        "unknown_action": "reject_with_error",
        "missing_or_blank_action": "reject_with_error",
        "binary_mapping": "0: Benign, 1: Attack",
        "multiclass_claimed": False,
    }
    eval_results["evaluation_method"] = "Scikit-Learn Binary Metrics on External Dataset"
    eval_results["data_overlap_note"] = (
        "Evaluated on user-supplied dataset. Note: Cross-dataset overlap with historical training partitions "
        "has not been determined and cannot be verified without dataset fingerprinting. No claim of independent "
        "evaluation is made if evaluation data overlaps with training data."
    )

    logger.info("Evaluation Summary:")
    logger.info("  Accuracy:            %.4f", eval_results["accuracy"])
    logger.info("  Precision:           %.4f", eval_results["precision"])
    logger.info("  Recall:              %.4f", eval_results["recall"])
    logger.info("  F1-Score:            %.4f", eval_results["f1_score"])
    logger.info("  False Positive Rate: %.4f", eval_results["false_positive_rate"])

    save_evaluation_report(eval_results, report_out, overwrite=args.overwrite)
    logger.info("Saved evaluation report to %s", report_out)

    return 0


def run_predict(args: argparse.Namespace, logger) -> int:
    """Execute batch prediction and alert generation workflow."""
    logger.info("Executing batch prediction workflow...")
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    pred_out = Path(args.predictions_output)
    alerts_out = Path(args.alerts_output)

    if not model_path.exists():
        logger.error("Model artifact not found at '%s'. Run in train mode first.", model_path)
        return 1

    if pred_out.exists() and not args.overwrite:
        logger.error(
            "Predictions output '%s' already exists. Use --overwrite to replace.", pred_out
        )
        return 1

    if alerts_out.exists() and not args.overwrite:
        logger.error(
            "Alerts output '%s' already exists. Use --overwrite to replace.", alerts_out
        )
        return 1

    logger.info("Loading model artifact from %s", model_path)
    pipeline, metadata = load_model_for_inference(model_path)

    schema = None
    if metadata.get("feature_schema"):
        schema = FeatureSchema.from_dict(metadata["feature_schema"])

    logger.info("Loading input flows from %s", data_path)
    df = load_dataset(data_path)
    df_clean = handle_non_finite_values(df)

    logger.info("Generating predictions and attack scores for %d flows...", len(df_clean))
    predictions, probabilities = predict_network_flows(
        model=pipeline,
        flow_data=df_clean,
        feature_schema=schema,
        pos_label=1,
    )

    # Build row-level predictions output DataFrame preserving original columns
    pred_df = df.copy()
    pred_df["predicted_class"] = predictions
    pred_df["predicted_label"] = [
        "Attack" if p == 1 or prob >= args.threshold else "Benign"
        for p, prob in zip(predictions, probabilities)
    ]
    pred_df["attack_score"] = [round(float(prob), 4) for prob in probabilities]

    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_out, index=False)
    logger.info("Saved row-level predictions to %s", pred_out)

    # Process and export structured security alerts
    logger.info("Evaluating alert thresholds (threshold=%.2f)...", args.threshold)
    alerts = process_batch_alerts(
        predictions=predictions,
        probabilities=probabilities,
        flow_data=df,
        threshold=args.threshold,
    )

    save_alerts_json(alerts, alerts_out, overwrite=args.overwrite)
    logger.info(
        "Generated %d alerts from %d total flows (saved to %s)",
        len(alerts),
        len(df),
        alerts_out,
    )

    return 0


def run_validate_data(args: argparse.Namespace, logger) -> int:
    """Execute raw dataset validation workflow."""
    logger.info("Executing dataset validation workflow...")
    data_dir = args.data_path if args.data_path else (args.data_dir or "data/raw")
    report_out = args.report_out or args.report_output or "outputs/reports/dataset_validation.json"
    chunksize = getattr(args, "chunksize", 10000)
    duplicate_check = getattr(args, "duplicate_check", "within_file")

    if chunksize <= 0:
        logger.error("Invalid chunksize: %d. chunksize must be a positive integer (> 0).", chunksize)
        return 1

    # Attempt to load model schema if model artifact exists
    schema = None
    model_path = Path(args.model_path)
    if model_path.exists():
        try:
            _, metadata = load_model_for_inference(model_path)
            if metadata.get("feature_schema"):
                schema = FeatureSchema.from_dict(metadata["feature_schema"])
            elif metadata.get("feature_names"):
                schema = FeatureSchema(
                    feature_names=metadata["feature_names"],
                    numeric_features=metadata.get("numeric_features", metadata["feature_names"]),
                )
            logger.info("Loaded model schema from %s for compatibility check.", model_path)
        except Exception as exc:
            logger.warning("Could not load model schema from %s: %s", model_path, exc)

    logger.info(
        "Validating CSV files in '%s' (chunksize=%d, duplicate_check=%s)...",
        data_dir,
        chunksize,
        duplicate_check,
    )
    try:
        results = validate_dataset_directory(
            data_dir=data_dir,
            report_out=report_out,
            chunksize=chunksize,
            model_schema=schema,
            duplicate_check=duplicate_check,
            overwrite=args.overwrite,
        )
    except FileExistsError as fee:
        logger.error("%s", fee)
        return 1

    status = results.get("status")
    total_files = results.get("total_files_discovered", 0)

    if status == "NO_FILES_FOUND":
        logger.error(
            "No CSV files found in '%s'. %s",
            data_dir,
            results.get("actionable_guidance", "Place dataset CSV files into data/raw/."),
        )
        return 1

    if status == "DIRECTORY_NOT_FOUND":
        logger.error("Data directory '%s' does not exist.", data_dir)
        return 1

    logger.info("Validation complete for %d files. Status: %s", total_files, status)
    logger.info("Report saved to: %s", report_out)

    agg = results.get("aggregate_summary", {})
    logger.info("  Total Rows:             %s", f"{agg.get('total_rows', 0):,}")
    logger.info("  Total Size:             %.2f MB", agg.get("total_file_size_mb", 0.0))
    logger.info("  Total Missing Values:   %s", f"{agg.get('total_missing_values', 0):,}")
    logger.info("  Total Positive Inf:     %s", f"{agg.get('total_positive_infinity', 0):,}")
    logger.info("  Total Negative Inf:     %s", f"{agg.get('total_negative_infinity', 0):,}")
    logger.info("  Duplicate Strategy:     %s", agg.get("duplicate_check_strategy", duplicate_check))
    logger.info("  Within-File Duplicates: %s", f"{agg.get('total_within_file_duplicates', 0):,}")
    logger.info("  Cross-File Duplicates:  %s", f"{agg.get('total_cross_file_duplicates', 0):,}")
    logger.info("  Total Duplicate Rows:   %s", f"{agg.get('total_duplicate_rows', 0):,}")
    logger.info("  Total Missing Labels:   %s", f"{agg.get('total_missing_labels', 0):,}")
    logger.info("  Total Blank Labels:     %s", f"{agg.get('total_blank_labels', 0):,}")
    logger.info("  Observed Labels:        %s", agg.get("aggregate_label_frequencies", {}))

    if agg.get("aggregate_unknown_labels"):
        logger.warning("  Unknown Labels Flagged: %s", agg.get("aggregate_unknown_labels"))

    if agg.get("common_leakage_risks"):
        logger.info("  Identified Leakage Risks:")
        for risk in agg.get("common_leakage_risks", []):
            logger.info("    - %s", risk)

    if status == "VALIDATION_ERRORS":
        logger.error("Validation encountered errors in one or more files.")
        return 1

    return 0


def run_experiment(args: argparse.Namespace, logger) -> int:
    """Execute Milestone 16 comparative experiments across models and captures."""
    logger.info("Executing Milestone 16 comparative model experiments...")
    train_paths_raw = [p.strip() for p in args.train_paths.split(",") if p.strip()]
    train_paths = [Path(p) for p in train_paths_raw]
    for p in train_paths:
        if not p.exists():
            logger.error("Training dataset capture not found: %s", p)
            return 1

    independent_test = Path(args.independent_test_path)
    if not independent_test.exists():
        logger.error("Independent test capture not found: %s", independent_test)
        return 1

    sample_size = args.max_samples if args.max_samples is not None else 100000
    report = run_milestone16_experiments(
        train_paths=train_paths,
        independent_test_path=independent_test,
        output_dir="outputs/reports",
        models_dir="models",
        sample_size=sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        overwrite=args.overwrite,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY RESULTS:")
    for m_id, m_data in report["model_comparison"].items():
        logger.info(
            "  Model: %-25s | Val Acc: %.4f | Val Rec: %.4f | Wed Rec: %.4f | Wed FPR: %.4f",
            m_id,
            m_data["validation_accuracy"],
            m_data["validation_recall"],
            m_data["wednesday_recall"],
            m_data["wednesday_fpr"],
        )
    logger.info("=" * 60)
    logger.info("Saved comparison report to outputs/reports/model_comparison_experiment.json")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for threat-detection CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    log_file = Path("outputs/logs/cli.log")
    logger = setup_logger("threat_detection_cli", log_file=log_file)

    mode = args.subcommand or args.mode
    if not mode:
        parser.error(
            "Must specify an operational mode (train, evaluate, predict, validate-data, experiment) "
            "via positional command or --mode flag."
        )

    # Validate that data-path or data-dir is supplied when required
    if mode in ["train", "evaluate"]:
        if not args.data_path and not args.data_dir:
            parser.error(f"--data-path or --data-dir is required for '{mode}' mode.")
        if not args.data_path:
            args.data_path = args.data_dir
    elif mode == "predict":
        if not args.data_path:
            parser.error(f"--data-path is required for '{mode}' mode.")

    try:
        if mode == "train":
            return run_train(args, logger)
        elif mode == "evaluate":
            return run_evaluate(args, logger)
        elif mode == "predict":
            return run_predict(args, logger)
        elif mode == "validate-data":
            return run_validate_data(args, logger)
        elif mode == "experiment":
            return run_experiment(args, logger)
        else:
            logger.error("Unknown mode: %s", mode)
            return 1
    except Exception as exc:
        logger.error("Execution failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

