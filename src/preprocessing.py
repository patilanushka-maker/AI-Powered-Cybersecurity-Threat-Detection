"""
Preprocessing Module for AI-Powered Cybersecurity Threat Detection System.

Provides reusable preprocessing utilities with explicit contracts:
- Column header normalization
- Safe conversion of positive and negative infinity to NaN for numeric columns
- Missing value and duplicate reporting
- Explicit duplicate row handling
- Target label validation and feature/target separation
- Leakage-aware train/test splitting (random stratified and temporal)
- Explicit binary label encoding (Benign -> 0, Attack -> 1) with strict whitelist enforcement
"""

from typing import Tuple, Dict, Any, Optional, Iterable, List, Set, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import normalize_column_names

# Explicit canonical whitelists for binary classification mapping
DEFAULT_BENIGN_LABELS: Set[str] = {"BENIGN", "0", "NORMAL"}
DEFAULT_RECOGNIZED_ATTACK_LABELS: Set[str] = {
    "ATTACK",
    "1",
    "MALICIOUS",
    "DDOS",
    "DOS",
    "PORTSCAN",
    "BOT",
    "INFILTRATION",
    "BRUTE FORCE",
    "WEB ATTACK",
    "WEB ATTACK - BRUTE FORCE",
    "WEB ATTACK – BRUTE FORCE",
    "WEB ATTACK - XSS",
    "WEB ATTACK – XSS",
    "WEB ATTACK - SQL INJECTION",
    "WEB ATTACK – SQL INJECTION",
    "HEARTBLEED",
    "FTP-PATATOR",
    "SSH-PATATOR",
    "DOS HULK",
    "DOS GOLDENEYE",
    "DOS SLOWLORIS",
    "DOS SLOWHTTPTEST",
    "DDOS LOIC",
}


def clean_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column headers by converting to strings and stripping whitespace.
    Reuses the canonical normalization logic from data_loader.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw column headers.

    Returns
    -------
    pd.DataFrame
        Shallow copy of DataFrame with stripped, verified column headers.

    Raises
    ------
    ValueError
        If duplicate column names occur after normalization.
    """
    df_copy = df.copy()
    df_copy.columns = normalize_column_names(df_copy.columns)
    return df_copy


def handle_non_finite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert positive and negative infinity in numerical columns to NaN (missing values),
    preserving string, object, and valid numerical columns intact.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame where numerical inf and -inf values are replaced with np.nan.
    """
    df_clean = df.copy()
    # Select exclusively numerical columns (integers and floats)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    return df_clean


def summarize_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Report detailed counts of missing values per column and duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to evaluate.

    Returns
    -------
    Dict[str, Any]
        Dictionary reporting total rows, total columns, missing values per column,
        total missing cells, and duplicate row count.
    """
    missing_per_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_per_column": missing_per_col,
        "total_missing_cells": total_missing,
        "duplicate_rows_count": duplicate_rows,
    }


def handle_duplicates(
    df: pd.DataFrame, keep: str = "first", remove: bool = False
) -> Tuple[pd.DataFrame, int]:
    """
    Inspect or optionally remove duplicate rows from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    keep : {'first', 'last', False}, default 'first'
        Determines which duplicates (if any) to keep when remove=True.
    remove : bool, default False
        If True, drop detected duplicate rows. If False, only report count
        and preserve original rows to prevent silent data loss.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        (result_dataframe, count_of_duplicate_rows)
    """
    duplicate_count = int(df.duplicated(keep=keep).sum())

    if remove and duplicate_count > 0:
        cleaned_df = df.drop_duplicates(keep=keep).copy()
        return cleaned_df, duplicate_count

    return df.copy(), duplicate_count


def validate_target_labels(
    labels: pd.Series,
    target_column: str = "Label",
    reject_missing: bool = True,
    allowed_labels: Optional[Iterable[str]] = None,
) -> pd.Series:
    """
    Validate target label series for missing values and unauthorized classes.

    Parameters
    ----------
    labels : pd.Series
        Series containing target class labels.
    target_column : str, default 'Label'
        Name of target column for diagnostic reporting.
    reject_missing : bool, default True
        If True, raise ValueError if any label is NaN or empty.
    allowed_labels : Optional[Iterable[str]], default None
        Optional whitelist of permissible label values. If provided, any label
        outside this set triggers a ValueError.

    Returns
    -------
    pd.Series
        Validated target labels.

    Raises
    ------
    ValueError
        If missing labels are detected (and reject_missing=True) or unexpected
        labels outside allowed_labels are present.
    """
    # Check for missing labels
    missing_count = int(labels.isna().sum())
    if missing_count > 0 and reject_missing:
        raise ValueError(
            f"Target column '{target_column}' contains {missing_count} missing or null values. "
            f"Silent imputation of target labels is disallowed."
        )

    # Check for empty string labels if labels are string-like
    if labels.dtype == object or isinstance(labels.dtype, pd.StringDtype):
        empty_str_mask = labels.astype(str).str.strip() == ""
        empty_str_count = int(empty_str_mask.sum())
        if empty_str_count > 0 and reject_missing:
            raise ValueError(
                f"Target column '{target_column}' contains {empty_str_count} empty string labels."
            )

    # Check allowed labels whitelist if provided
    if allowed_labels is not None:
        allowed_set: Set[str] = set(str(lbl).strip().upper() for lbl in allowed_labels)
        unexpected = [lbl for lbl in labels.dropna().unique() if str(lbl).strip().upper() not in allowed_set]
        if unexpected:
            raise ValueError(
                f"Target column '{target_column}' contains unexpected labels: {sorted(list(unexpected))}. "
                f"Allowed labels: {sorted(list(allowed_labels))}."
            )

    return labels


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = "Label",
    reject_missing: bool = True,
    allowed_labels: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix from target label series with explicit validation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with features and target.
    target_column : str, default 'Label'
        Name of target column.
    reject_missing : bool, default True
        Whether to reject rows with missing labels.
    allowed_labels : Optional[Iterable[str]], default None
        Optional list of permissible label strings.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Tuple of (X, y) where X contains all features preserving original column
        order minus the target, and y contains validated target labels.

    Raises
    ------
    KeyError
        If target_column is not present in df.columns.
    ValueError
        If target column fails validation checks.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}"
        )

    # Extract target series and validate
    y = df[target_column].copy()
    validate_target_labels(
        y,
        target_column=target_column,
        reject_missing=reject_missing,
        allowed_labels=allowed_labels,
    )

    # Extract feature matrix without target column
    X = df.drop(columns=[target_column]).copy()

    # Integrity assertions: target must not bleed into X, row count must match
    assert target_column not in X.columns, f"Target '{target_column}' leaked into feature matrix"
    assert len(X) == len(y), f"Row count mismatch between X ({len(X)}) and y ({len(y)})"

    return X, y


def encode_binary_target(
    y: pd.Series,
    benign_label: Optional[str] = None,
    benign_labels: Optional[Iterable[str]] = None,
    attack_labels: Optional[Iterable[str]] = None,
    reject_unknown: bool = True,
) -> pd.Series:
    """
    Map target labels into binary integers: 0 for Benign, 1 for Attack.

    Label Policy:
    - Normalizes strings by trimming whitespace and converting to uppercase.
    - Matches recognized benign labels to 0.
    - Matches recognized attack labels to 1.
    - Rejects missing, empty, and unexpected labels under reject_unknown=True.

    Parameters
    ----------
    y : pd.Series
        Series containing target labels.
    benign_label : Optional[str], default None
        Single string for benign label (for backwards compatibility).
    benign_labels : Optional[Iterable[str]], default None
        Iterable of strings denoting benign traffic. Defaults to DEFAULT_BENIGN_LABELS.
    attack_labels : Optional[Iterable[str]], default None
        Iterable of strings denoting recognized attack traffic.
        Defaults to DEFAULT_RECOGNIZED_ATTACK_LABELS.
    reject_unknown : bool, default True
        If True, raises ValueError if any label is neither in benign nor in attack sets.

    Returns
    -------
    pd.Series
        Integer Series with 0 (benign) and 1 (attack).

    Raises
    ------
    ValueError
        If missing or empty labels are found, or if unknown labels appear when reject_unknown=True.
    """
    if y.isna().any():
        raise ValueError("Cannot encode binary target: Series contains null or missing values.")

    # Strip whitespace, normalize dashes, and convert to string for uniform comparison
    y_str = (
        y.astype(str)
        .str.strip()
        .str.replace("\u2013", "-", regex=False)
        .str.replace("\u2014", "-", regex=False)
    )

    if (y_str == "").any():
        raise ValueError("Cannot encode binary target: Series contains empty string labels.")

    # Build benign set
    benign_set: Set[str] = set()
    if benign_labels is not None:
        benign_set.update(str(b).strip().upper() for b in benign_labels)
    elif benign_label is not None:
        benign_set.add(str(benign_label).strip().upper())
    else:
        benign_set.update(DEFAULT_BENIGN_LABELS)

    if benign_label is not None:
        benign_set.add(str(benign_label).strip().upper())

    # Build attack set
    attack_set: Set[str] = set()
    if attack_labels is not None:
        attack_set.update(str(a).strip().upper() for a in attack_labels)
    else:
        attack_set.update(DEFAULT_RECOGNIZED_ATTACK_LABELS)

    # Intersection safety check
    overlap = benign_set.intersection(attack_set)
    if overlap:
        raise ValueError(
            f"Label conflict: Labels {overlap} are defined as both benign and attack."
        )

    y_upper = y_str.str.upper()
    benign_mask = y_upper.isin(benign_set)
    attack_mask = y_upper.isin(attack_set)

    unrecognized_mask = ~benign_mask & ~attack_mask
    if unrecognized_mask.any():
        unknowns = sorted(list(y_str[unrecognized_mask].unique()))
        if reject_unknown:
            raise ValueError(
                f"Unknown or unexpected label(s) detected during binary encoding: {unknowns}. "
                f"Expected recognized benign {sorted(list(benign_set))} or recognized attack {sorted(list(attack_set))}."
            )

    binary_y = pd.Series(index=y.index, dtype=np.int64)
    binary_y[benign_mask] = 0
    binary_y[attack_mask] = 1
    if not reject_unknown and unrecognized_mask.any():
        binary_y[unrecognized_mask] = 1

    return binary_y


def train_test_split_network_flows(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition network flow features and labels into train and test splits before
    any data-dependent transformers are fitted.

    Note on Data Leakage:
    Stratification ensures balanced class proportions, but does NOT prevent leakage
    caused by identical flow records, session re-transmissions, or co-occurring traffic
    collected at the exact same second. Ensure duplicate removal and identifier exclusion
    are applied beforehand.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (or binary encoded labels).
    test_size : float, default 0.2
        Fraction of samples allocated to test split.
    random_state : int, default 42
        Reproducibility seed.
    stratify : bool, default True
        Whether to perform stratified sampling across class labels.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)

    Raises
    ------
    ValueError
        If X and y length mismatch, dataset contains fewer than 2 classes,
        or class counts are insufficient for stratified splitting.
    """
    if len(X) != len(y):
        raise ValueError(
            f"Length mismatch: X has {len(X)} rows, but y has {len(y)} rows."
        )

    if len(X) < 4:
        raise ValueError(
            f"Dataset is too small for train/test splitting ({len(X)} rows)."
        )

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 distinct classes for evaluation. "
            f"Found classes: {list(unique_classes)}"
        )

    class_counts = y.value_counts()
    min_count = class_counts.min()

    stratify_target = None
    if stratify:
        if min_count < 2:
            raise ValueError(
                f"Cannot stratify split: minority class has fewer than 2 samples. "
                f"Class counts: {class_counts.to_dict()}"
            )
        stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test


def split_data_temporal(
    df: pd.DataFrame,
    time_column: str,
    split_ratio: float = 0.8,
    target_column: str = "Label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a genuine temporal train/test split by parsing timestamps and
    sorting chronologically.

    Note on Data Leakage:
    Temporal splitting reflects realistic forward-in-time deployment, but does not
    automatically prevent host-level, session-level, or duplicate-flow leakage across
    split boundaries if ongoing communication spans the split threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features, target, and a timestamp column.
    time_column : str
        Name of timestamp/datetime column.
    split_ratio : float, default 0.8
        Proportion of earliest records allocated to train split (0.0 < split_ratio < 1.0).
    target_column : str, default 'Label'
        Target label column name.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)

    Raises
    ------
    KeyError
        If time_column or target_column is not in df.columns.
    ValueError
        If split_ratio is invalid, dataset has insufficient rows (< 4),
        or timestamps are invalid or unparseable.
    """
    if time_column not in df.columns:
        raise KeyError(f"Timestamp column '{time_column}' not found in DataFrame.")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    if len(df) < 4:
        raise ValueError(
            f"Dataset has insufficient rows for temporal splitting ({len(df)} rows, minimum 4 required)."
        )

    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be between 0.0 and 1.0, got {split_ratio}.")

    # Validate and parse timestamps to datetime
    parsed_times = pd.to_datetime(df[time_column], errors="coerce")
    invalid_count = int(parsed_times.isna().sum())
    if invalid_count > 0:
        raise ValueError(
            f"Timestamp column '{time_column}' contains {invalid_count} invalid or missing timestamps. "
            f"Temporal splitting requires valid chronologically parseable datetime values."
        )

    # Sort strictly chronologically by parsed timestamp
    df_sorted = df.iloc[parsed_times.argsort(kind="mergesort")].copy()

    split_idx = int(len(df_sorted) * split_ratio)
    if split_idx <= 0 or split_idx >= len(df_sorted):
        raise ValueError(
            f"split_ratio {split_ratio} produced an empty partition with {len(df_sorted)} total rows."
        )

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    # Drop target and timestamp from feature sets
    X_train = train_df.drop(columns=[target_column, time_column]).copy()
    X_test = test_df.drop(columns=[target_column, time_column]).copy()

    # Target leakage assertion
    assert target_column not in X_train.columns
    assert target_column not in X_test.columns

    return X_train, X_test, y_train, y_test


def sample_dataset(
    df: pd.DataFrame,
    max_samples: Optional[int] = None,
    target_column: Optional[str] = "Label",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministically and reproducibly sample a dataset to a maximum sample count.

    If target_column is present in df and classes have sufficient samples (>= 2),
    performs stratified sampling to preserve relative class distributions.
    Otherwise, performs reproducible uniform sampling.

    If max_samples is None or >= len(df), returns a copy of df without downsampling.
    Rows are NEVER silently discarded.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to optionally subsample.
    max_samples : Optional[int], default None
        Maximum number of rows to retain. If None or >= len(df), all rows are kept.
    target_column : Optional[str], default 'Label'
        Column name to stratify across, if available.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (sampled_df, sampling_metadata)

    Raises
    ------
    ValueError
        If max_samples <= 0.
    """
    total_rows = len(df)
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            f"max_samples must be a positive integer (> 0), got {max_samples}"
        )

    counts_before: Dict[str, int] = {}
    if target_column is not None and target_column in df.columns:
        counts_before = {str(k): int(v) for k, v in df[target_column].value_counts().items()}

    if max_samples is not None and target_column is not None and target_column in df.columns:
        num_classes = len(counts_before)
        if max_samples < num_classes:
            raise ValueError(
                f"max_samples ({max_samples}) cannot be smaller than the number of unique classes ({num_classes})."
            )

    if max_samples is None or max_samples >= total_rows:
        return df.copy(), {
            "sampling_applied": False,
            "total_rows_available": total_rows,
            "rows_retained": total_rows,
            "sampling_ratio": 1.0,
            "sampling_method": "none",
            "random_state": random_state,
            "class_distribution_before": counts_before,
            "class_distribution_after": counts_before,
            "classes_preserved": True,
        }

    # Attempt stratified sampling if target_column exists and classes allow it
    can_stratify = False
    if target_column is not None and target_column in df.columns:
        counts = df[target_column].value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            can_stratify = True

    if can_stratify:
        try:
            sampled_df, _ = train_test_split(
                df,
                train_size=max_samples,
                stratify=df[target_column],
                random_state=random_state,
            )
            method = "stratified"
        except Exception:
            sampled_df = df.sample(n=max_samples, random_state=random_state)
            method = "uniform_random"
    else:
        sampled_df = df.sample(n=max_samples, random_state=random_state)
        method = "uniform_random"

    # Guarantee that all classes present in df are represented in sampled_df
    # ensuring at least min(orig_count, 2) samples so downstream train/test splitting does not fail
    if target_column is not None and target_column in df.columns:
        sampled_indices = set(sampled_df.index)
        counts_after = {str(k): int(v) for k, v in sampled_df[target_column].value_counts().items()}

        missing_or_underrepresented = {}
        for cls_name, orig_count in counts_before.items():
            curr_count = counts_after.get(cls_name, 0)
            required = min(orig_count, 2)
            if curr_count < required:
                missing_or_underrepresented[cls_name] = required - curr_count

        if missing_or_underrepresented:
            sampled_df = sampled_df.copy()
            for cls_name, needed in missing_or_underrepresented.items():
                candidate_pool = df[
                    (df[target_column].astype(str) == cls_name) & (~df.index.isin(sampled_indices))
                ]
                if len(candidate_pool) > 0:
                    add_rows = candidate_pool.sample(
                        n=min(needed, len(candidate_pool)), random_state=random_state
                    )
                    # Find majority class rows to replace
                    maj_class = sampled_df[target_column].astype(str).value_counts().index[0]
                    maj_rows = sampled_df[sampled_df[target_column].astype(str) == maj_class]
                    if len(maj_rows) > len(add_rows):
                        drop_indices = maj_rows.index[: len(add_rows)]
                        sampled_df = sampled_df.drop(index=drop_indices)
                        sampled_df = pd.concat([sampled_df, add_rows])
                        sampled_indices = set(sampled_df.index)

    # Reset index for clean downstream handling
    sampled_df = sampled_df.reset_index(drop=True)

    counts_final: Dict[str, int] = {}
    classes_preserved = True
    if target_column is not None and target_column in sampled_df.columns:
        counts_final = {str(k): int(v) for k, v in sampled_df[target_column].value_counts().items()}
        classes_preserved = set(counts_final.keys()) == set(counts_before.keys())

    metadata: Dict[str, Any] = {
        "sampling_applied": True,
        "total_rows_available": total_rows,
        "rows_retained": len(sampled_df),
        "sampling_ratio": round(len(sampled_df) / total_rows, 4),
        "sampling_method": method,
        "random_state": random_state,
        "class_distribution_before": counts_before,
        "class_distribution_after": counts_final,
        "classes_preserved": classes_preserved,
    }

    return sampled_df, metadata
