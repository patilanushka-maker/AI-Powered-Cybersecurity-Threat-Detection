"""
Dataset Validation Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Memory-aware, chunked inspection of network flow CSV datasets in data/raw/
- Verifying file sizes, row counts, column naming, missing values, and infinities
- Profiling label columns, label frequencies, blank/missing labels, and unknown/unrecognized labels
- Parsing timestamp columns and quantifying timestamp conversion failures
- Auditing schema compatibility against trained FeatureSchema definitions
- Reviewing target and session leakage risks (identifiers, raw timestamps, duplicates)
- Configurable duplicate row detection (within-file, cross-file, or none for constant-memory)
- Generating comprehensive, auditable JSON validation reports WITHOUT performing model training
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
import numpy as np
import pandas as pd

from src.data_loader import normalize_column_names, list_raw_datasets
from src.preprocessing import DEFAULT_BENIGN_LABELS, DEFAULT_RECOGNIZED_ATTACK_LABELS
from src.feature_engineering import FeatureSchema, DEFAULT_METADATA_COLUMNS


def get_relative_path_str(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Sanitize and relativize a path to prevent exposing sensitive local absolute directories.

    Parameters
    ----------
    path : Union[str, Path]
        Path to format.
    base_dir : Optional[Union[str, Path]], default None
        Optional base directory. If None, uses Path.cwd().

    Returns
    -------
    str
        Relative path formatted with forward slashes.
    """
    p = Path(path)
    base = Path(base_dir) if base_dir else Path.cwd()
    try:
        rel = p.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def validate_single_csv(
    file_path: Union[str, Path],
    chunksize: int = 10000,
    model_schema: Optional[FeatureSchema] = None,
    duplicate_check: str = "within_file",
    seen_global_hashes: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    """
    Perform a memory-safe, chunked audit of an individual network flow CSV file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to CSV file.
    chunksize : int, default 10000
        Number of rows to process per chunk. Must be > 0.
    model_schema : Optional[FeatureSchema], default None
        Optional model schema to evaluate compatibility against.
    duplicate_check : str, default 'within_file'
        Strategy for duplicate row tracking: 'within_file', 'cross_file', or 'none'.
    seen_global_hashes : Optional[Set[int]], default None
        Shared set of 64-bit integer hashes from previously processed files
        (used when duplicate_check='cross_file').

    Returns
    -------
    Dict[str, Any]
        Structured validation results for the file.

    Raises
    ------
    ValueError
        If chunksize <= 0 or duplicate_check is invalid.
    """
    if chunksize <= 0:
        raise ValueError(f"chunksize must be a positive integer (> 0), got {chunksize}")

    valid_dup_strategies = {"within_file", "cross_file", "none"}
    if duplicate_check not in valid_dup_strategies:
        raise ValueError(
            f"Invalid duplicate_check strategy '{duplicate_check}'. Must be one of: {sorted(list(valid_dup_strategies))}"
        )

    path = Path(file_path)
    rel_path_str = get_relative_path_str(path)

    if not path.exists():
        return {
            "file_name": path.name,
            "relative_path": rel_path_str,
            "status": "ERROR",
            "error_type": "FileNotFoundError",
            "error_message": f"File does not exist: '{path.name}'",
        }

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    if file_size_bytes == 0:
        return {
            "file_name": path.name,
            "relative_path": rel_path_str,
            "file_size_bytes": 0,
            "file_size_mb": 0.0,
            "status": "ERROR",
            "error_type": "EmptyFileError",
            "error_message": f"File '{path.name}' is empty (0 bytes).",
        }

    # Attempt to open and inspect chunks
    try:
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            low_memory=False,
            encoding_errors="replace",
        )
    except Exception as exc:
        return {
            "file_name": path.name,
            "relative_path": rel_path_str,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": f"Failed to initialize CSV reader for '{path.name}': {exc}",
        }

    total_rows = 0
    raw_columns: List[str] = []
    normalized_columns: List[str] = []
    column_dtypes: Dict[str, str] = {}
    missing_counts: Dict[str, int] = {}
    pos_inf_counts: Dict[str, int] = {}
    neg_inf_counts: Dict[str, int] = {}

    label_candidates: List[str] = []
    label_frequencies: Dict[str, int] = {}
    unknown_labels: Set[str] = set()
    missing_labels_count: int = 0
    blank_labels_count: int = 0

    timestamp_candidates: List[str] = []
    timestamp_parsing_failures: int = 0
    timestamp_missing_count: int = 0

    seen_file_hashes: Set[int] = set()
    within_file_duplicate_rows_count: int = 0
    cross_file_duplicate_rows_count: int = 0

    first_chunk = True

    try:
        for chunk in reader:
            if first_chunk:
                raw_columns = [str(c) for c in chunk.columns]
                try:
                    normalized_columns = normalize_column_names(chunk.columns)
                except ValueError as norm_err:
                    return {
                        "file_name": path.name,
                        "relative_path": rel_path_str,
                        "file_size_bytes": file_size_bytes,
                        "file_size_mb": file_size_mb,
                        "status": "ERROR",
                        "error_type": "DuplicateColumnsError",
                        "error_message": str(norm_err),
                    }

                chunk.columns = normalized_columns
                for col in normalized_columns:
                    column_dtypes[col] = str(chunk[col].dtype)
                    missing_counts[col] = 0
                    pos_inf_counts[col] = 0
                    neg_inf_counts[col] = 0

                # Detect candidate columns
                for col in normalized_columns:
                    col_lower = col.lower()
                    if col_lower in ["label", "class", "attack"]:
                        label_candidates.append(col)
                    elif "label" in col_lower or "class" in col_lower:
                        label_candidates.append(col)

                    if col_lower in ["timestamp", "date", "time"] or "time" in col_lower:
                        timestamp_candidates.append(col)

                first_chunk = False
            else:
                chunk.columns = normalized_columns

            chunk_len = len(chunk)
            total_rows += chunk_len

            if chunk_len == 0:
                continue

            # Missing values and Infinities per column
            for col in normalized_columns:
                series = chunk[col]
                missing_counts[col] += int(series.isna().sum())

                if pd.api.types.is_numeric_dtype(series):
                    pos_inf_counts[col] += int((series == np.inf).sum())
                    neg_inf_counts[col] += int((series == -np.inf).sum())

            # Label profiling
            if label_candidates:
                primary_label_col = label_candidates[0]
                raw_lbl = chunk[primary_label_col]
                missing_labels_count += int(raw_lbl.isna().sum())

                non_null_lbl = raw_lbl.dropna().astype(str)
                stripped_lbl = non_null_lbl.str.strip()
                blank_labels_count += int((stripped_lbl == "").sum())

                non_blank_lbl = stripped_lbl[stripped_lbl != ""]
                for label_val, count in non_blank_lbl.value_counts().items():
                    label_frequencies[label_val] = label_frequencies.get(label_val, 0) + int(count)

                    # Check against canonical recognized vocabulary
                    norm_val = str(label_val).upper()
                    if (
                        norm_val not in DEFAULT_BENIGN_LABELS
                        and norm_val not in DEFAULT_RECOGNIZED_ATTACK_LABELS
                    ):
                        unknown_labels.add(str(label_val))

            # Timestamp parsing verification
            if timestamp_candidates:
                primary_ts_col = timestamp_candidates[0]
                ts_series = chunk[primary_ts_col]
                timestamp_missing_count += int(ts_series.isna().sum())

                non_null_ts = ts_series.dropna().astype(str).str.strip()
                non_empty_ts = non_null_ts[non_null_ts != ""]
                if len(non_empty_ts) > 0:
                    parsed = pd.to_datetime(non_empty_ts, format="mixed", errors="coerce")
                    failures = int(parsed.isna().sum())
                    timestamp_parsing_failures += failures

            # Memory-aware duplicate row tracking using 64-bit integer hashes
            if duplicate_check != "none":
                row_hashes = pd.util.hash_pandas_object(chunk, index=False).values
                for h in row_hashes:
                    if h in seen_file_hashes:
                        within_file_duplicate_rows_count += 1
                    elif duplicate_check == "cross_file" and seen_global_hashes is not None and h in seen_global_hashes:
                        cross_file_duplicate_rows_count += 1
                        seen_file_hashes.add(h)
                    else:
                        seen_file_hashes.add(h)
                        if duplicate_check == "cross_file" and seen_global_hashes is not None:
                            seen_global_hashes.add(h)

    except Exception as exc:
        return {
            "file_name": path.name,
            "relative_path": rel_path_str,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message": f"Error occurred during chunked reading of '{path.name}': {exc}",
        }

    # Total duplicates for this file
    total_file_duplicates = within_file_duplicate_rows_count + cross_file_duplicate_rows_count

    # Derive percentages and summaries
    label_percentages: Dict[str, float] = {}
    if total_rows > 0 and label_frequencies:
        total_labeled = sum(label_frequencies.values())
        for lbl, count in label_frequencies.items():
            label_percentages[lbl] = round((count / total_labeled) * 100.0, 3)

    all_missing_cols = [col for col, count in missing_counts.items() if count == total_rows and total_rows > 0]
    non_numeric_cols = [
        col for col, dtype in column_dtypes.items() if not dtype.startswith(("int", "float"))
    ]

    # Schema compatibility assessment
    schema_compat: Dict[str, Any] = {
        "model_schema_provided": model_schema is not None,
        "is_compatible": True,
        "missing_required_features": [],
        "extra_columns_count": 0,
        "target_present": False,
    }

    if model_schema is not None:
        expected = set(model_schema.feature_names)
        observed = set(normalized_columns)
        missing_req = sorted(list(expected - observed))
        extra_cols = sorted(list(observed - expected))

        target_in_df = model_schema.target_column in normalized_columns
        is_compat = (len(missing_req) == 0) and (target_in_df or model_schema.target_column is None)

        schema_compat["is_compatible"] = is_compat
        schema_compat["missing_required_features"] = missing_req
        schema_compat["extra_columns_count"] = len(extra_cols)
        schema_compat["target_present"] = target_in_df

    # Leakage Risk Analysis
    leakage_risks: List[str] = []
    observed_metadata = [col for col in normalized_columns if col in DEFAULT_METADATA_COLUMNS]
    if observed_metadata:
        leakage_risks.append(
            f"Identifier/metadata columns detected ({observed_metadata}). "
            "Must be excluded from feature matrix to prevent memorization leakage."
        )

    if timestamp_candidates:
        leakage_risks.append(
            f"Timestamp column detected ('{timestamp_candidates[0]}'). "
            "Random stratified splitting across time series risks temporal leakage. "
            "Note: Temporal splitting orders chronologically but does not prevent duplicate-flow or host-session leakage."
        )

    if total_file_duplicates > 0:
        dup_pct = round((total_file_duplicates / max(total_rows, 1)) * 100, 2)
        leakage_risks.append(
            f"Identified {total_file_duplicates} duplicate rows ({dup_pct}% of file). "
            f"[Within-file: {within_file_duplicate_rows_count}, Cross-file: {cross_file_duplicate_rows_count}]. "
            "Identical flows spanning train and test partitions cause optimistic performance estimation."
        )

    if missing_labels_count > 0 or blank_labels_count > 0:
        leakage_risks.append(
            f"Target column contains {missing_labels_count} missing (NaN) and {blank_labels_count} blank label(s). "
            "Rows with missing or blank targets must be addressed before supervised training."
        )

    status = "SUCCESS"
    if (
        total_rows == 0
        or unknown_labels
        or all_missing_cols
        or missing_labels_count > 0
        or blank_labels_count > 0
        or (model_schema and not schema_compat["is_compatible"])
    ):
        status = "VALIDATION_WARNINGS"

    dup_note = (
        "Duplicate detection uses 64-bit row hashes (pd.util.hash_pandas_object). "
        "64-bit hash equality is a probabilistic duplicate estimate, not mathematical proof of identical contents. "
        "Hash collisions could theoretically cause a slight overcounting."
        if duplicate_check != "none"
        else "Duplicate detection was disabled (duplicate_check='none')."
    )

    return {
        "file_name": path.name,
        "relative_path": rel_path_str,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "status": status,
        "total_rows": total_rows,
        "column_count": len(normalized_columns),
        "normalized_columns": normalized_columns,
        "column_dtypes": column_dtypes,
        "missing_values_per_column": missing_counts,
        "total_missing_values": sum(missing_counts.values()),
        "positive_inf_counts": pos_inf_counts,
        "total_positive_inf": sum(pos_inf_counts.values()),
        "negative_inf_counts": neg_inf_counts,
        "total_negative_inf": sum(neg_inf_counts.values()),
        "duplicate_check_strategy": duplicate_check,
        "duplicate_rows_count": total_file_duplicates,
        "within_file_duplicate_rows_count": within_file_duplicate_rows_count,
        "cross_file_duplicate_rows_count": cross_file_duplicate_rows_count,
        "duplicate_detection_note": dup_note,
        "all_missing_columns": all_missing_cols,
        "non_numeric_columns": non_numeric_cols,
        "label_candidates": label_candidates,
        "missing_labels_count": missing_labels_count,
        "blank_labels_count": blank_labels_count,
        "label_frequencies": label_frequencies,
        "label_percentages": label_percentages,
        "unknown_labels": sorted(list(unknown_labels)),
        "timestamp_candidates": timestamp_candidates,
        "timestamp_parsing_failures": timestamp_parsing_failures,
        "timestamp_missing_count": timestamp_missing_count,
        "schema_compatibility": schema_compat,
        "leakage_risks": leakage_risks,
    }


def validate_dataset_directory(
    data_dir: Union[str, Path] = "data/raw",
    report_out: Optional[Union[str, Path]] = None,
    chunksize: int = 10000,
    model_schema: Optional[FeatureSchema] = None,
    duplicate_check: str = "within_file",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Validate all raw CSV datasets in the specified directory.

    Parameters
    ----------
    data_dir : Union[str, Path], default 'data/raw'
        Directory containing raw network flow CSV files.
    report_out : Optional[Union[str, Path]], default None
        Destination path for JSON report.
    chunksize : int, default 10000
        Number of rows per processing chunk. Must be > 0.
    model_schema : Optional[FeatureSchema], default None
        Optional model schema for compatibility checks.
    duplicate_check : str, default 'within_file'
        Strategy for duplicate row tracking: 'within_file', 'cross_file', or 'none'.
    overwrite : bool, default False
        Whether to overwrite an existing report file.

    Returns
    -------
    Dict[str, Any]
        Aggregated validation report.

    Raises
    ------
    ValueError
        If chunksize <= 0 or duplicate_check is invalid.
    """
    if chunksize <= 0:
        raise ValueError(f"chunksize must be a positive integer (> 0), got {chunksize}")

    valid_dup_strategies = {"within_file", "cross_file", "none"}
    if duplicate_check not in valid_dup_strategies:
        raise ValueError(
            f"Invalid duplicate_check strategy '{duplicate_check}'. Must be one of: {sorted(list(valid_dup_strategies))}"
        )

    dir_path = Path(data_dir)
    sanitized_dir = get_relative_path_str(dir_path)

    base_report: Dict[str, Any] = {
        "report_timestamp": datetime.utcnow().isoformat(),
        "validator_version": "1.1.0",
        "data_directory": sanitized_dir,
        "model_training_performed": False,
        "model_evaluation_performed": False,
        "chunksize_used": chunksize,
        "duplicate_check_strategy": duplicate_check,
        "runtime_notes": (
            "Chunked inspection reads data in batches of configured chunksize. "
            "Model training and performance evaluation were explicitly NOT performed."
        ),
    }

    if not dir_path.exists():
        base_report["status"] = "DIRECTORY_NOT_FOUND"
        base_report["total_files_discovered"] = 0
        base_report["error_message"] = f"Directory not found: '{sanitized_dir}'"
        base_report["files"] = []
        base_report["aggregate_summary"] = {}

        if report_out:
            _save_report(base_report, report_out, overwrite=overwrite)
        return base_report

    discovered_files = list_raw_datasets(dir_path)

    if not discovered_files:
        base_report["status"] = "NO_FILES_FOUND"
        base_report["total_files_discovered"] = 0
        base_report["actionable_guidance"] = (
            f"No CSV dataset files found in '{sanitized_dir}'. "
            "To validate CIC-IDS2017, download the official CSV files from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and place them into data/raw/."
        )
        base_report["files"] = []
        base_report["aggregate_summary"] = {}

        if report_out:
            _save_report(base_report, report_out, overwrite=overwrite)
        return base_report

    # Shared global hash set if cross_file duplicate checking is enabled
    seen_global_hashes: Optional[Set[int]] = set() if duplicate_check == "cross_file" else None

    # Inspect each discovered CSV file
    file_reports: List[Dict[str, Any]] = []
    total_rows_agg = 0
    total_size_bytes_agg = 0
    total_missing_agg = 0
    total_pos_inf_agg = 0
    total_neg_inf_agg = 0
    total_within_file_dups_agg = 0
    total_cross_file_dups_agg = 0
    total_missing_labels_agg = 0
    total_blank_labels_agg = 0
    aggregate_labels: Dict[str, int] = {}
    aggregate_unknown_labels: Set[str] = set()
    aggregate_leakage_risks: Set[str] = set()
    has_errors = False
    has_warnings = False

    for file_path in discovered_files:
        res = validate_single_csv(
            file_path,
            chunksize=chunksize,
            model_schema=model_schema,
            duplicate_check=duplicate_check,
            seen_global_hashes=seen_global_hashes,
        )
        file_reports.append(res)

        if res.get("status") == "ERROR":
            has_errors = True
            continue

        if res.get("status") == "VALIDATION_WARNINGS":
            has_warnings = True

        total_rows_agg += res.get("total_rows", 0)
        total_size_bytes_agg += res.get("file_size_bytes", 0)
        total_missing_agg += res.get("total_missing_values", 0)
        total_pos_inf_agg += res.get("total_positive_inf", 0)
        total_neg_inf_agg += res.get("total_negative_inf", 0)
        total_within_file_dups_agg += res.get("within_file_duplicate_rows_count", 0)
        total_cross_file_dups_agg += res.get("cross_file_duplicate_rows_count", 0)
        total_missing_labels_agg += res.get("missing_labels_count", 0)
        total_blank_labels_agg += res.get("blank_labels_count", 0)

        for lbl, count in res.get("label_frequencies", {}).items():
            aggregate_labels[lbl] = aggregate_labels.get(lbl, 0) + count

        for unk in res.get("unknown_labels", []):
            aggregate_unknown_labels.add(unk)

        for risk in res.get("leakage_risks", []):
            aggregate_leakage_risks.add(risk)

    overall_status = "SUCCESS"
    if has_errors:
        overall_status = "VALIDATION_ERRORS"
    elif has_warnings:
        overall_status = "VALIDATION_WARNINGS"

    dup_hash_note = (
        "Duplicate detection uses 64-bit row hashes (pd.util.hash_pandas_object). "
        "64-bit hash equality is a probabilistic duplicate estimate, not mathematical proof of identical contents. "
        "Hash collisions could theoretically cause a slight overcounting. "
        "When duplicate_check='none', duplicate tracking is skipped for true constant-memory execution."
        if duplicate_check != "none"
        else "Duplicate detection was disabled (duplicate_check='none')."
    )

    memory_assessment = (
        "Memory complexity: Tabular data is processed in chunks of O(chunksize * columns). "
        f"Duplicate detection strategy is '{duplicate_check}': "
        + (
            "Space is O(unique_rows_per_file) 64-bit integer hashes (~32-64 MB RAM per million unique rows, freed between files)."
            if duplicate_check == "within_file"
            else (
                "Space is O(total_unique_rows) 64-bit integer hashes retained across all files (~80-160 MB RAM for 2.8M rows)."
                if duplicate_check == "cross_file"
                else "No hash sets allocated; memory is strictly O(chunksize * columns) constant memory."
            )
        )
    )

    base_report["status"] = overall_status
    base_report["total_files_discovered"] = len(discovered_files)
    base_report["files"] = file_reports
    base_report["aggregate_summary"] = {
        "total_rows": total_rows_agg,
        "total_file_size_bytes": total_size_bytes_agg,
        "total_file_size_mb": round(total_size_bytes_agg / (1024 * 1024), 3),
        "total_missing_values": total_missing_agg,
        "total_positive_infinity": total_pos_inf_agg,
        "total_negative_infinity": total_neg_inf_agg,
        "duplicate_check_strategy": duplicate_check,
        "total_within_file_duplicates": total_within_file_dups_agg,
        "total_cross_file_duplicates": total_cross_file_dups_agg,
        "total_duplicate_rows": total_within_file_dups_agg + total_cross_file_dups_agg,
        "total_missing_labels": total_missing_labels_agg,
        "total_blank_labels": total_blank_labels_agg,
        "aggregate_label_frequencies": aggregate_labels,
        "aggregate_unknown_labels": sorted(list(aggregate_unknown_labels)),
        "common_leakage_risks": sorted(list(aggregate_leakage_risks)),
        "duplicate_hash_guarantee_note": dup_hash_note,
        "memory_complexity_assessment": memory_assessment,
    }

    if report_out:
        _save_report(base_report, report_out, overwrite=overwrite)

    return base_report


def _save_report(
    report: Dict[str, Any], output_path: Union[str, Path], overwrite: bool = False
) -> Path:
    """Save report dictionary to JSON with overwrite protection."""
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Validation report already exists at '{path}'. Pass overwrite=True to replace."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
