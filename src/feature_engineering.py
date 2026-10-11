"""
Feature Engineering Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for:
- Reusable FeatureSchema definitions for training and inference consistency
- Leakage-aware feature/target separation
- Explicit exclusion of identifier/timestamp metadata fields
- Non-numeric feature validation (rejecting silent coercion)
- Preserving column order, detecting duplicate column names, and rejecting all-missing columns
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable, Set, Dict, Any
import numpy as np
import pandas as pd
from src.preprocessing import split_features_and_target

# Standard network metadata and identifier columns that risk target leakage
# or memorization (e.g., specific IP addresses or session timestamps)
DEFAULT_METADATA_COLUMNS: List[str] = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Src IP",
    "Dst IP",
    "Date",
    "Time",
    "Capture Source",
    "capture_source",
]


@dataclass
class FeatureSchema:
    """
    Schema capturing expected feature names and types for inference validation.

    Attributes
    ----------
    feature_names : List[str]
        Ordered list of feature column names.
    numeric_features : List[str]
        List of numeric feature column names.
    metadata_excluded : List[str]
        List of excluded identifier/metadata columns.
    target_column : str
        Name of target column.
    """

    feature_names: List[str]
    numeric_features: List[str]
    metadata_excluded: List[str] = field(default_factory=list)
    target_column: str = "Label"

    def validate(
        self,
        df: pd.DataFrame,
        allow_extra_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Validate an input DataFrame against this schema and align feature order.

        Parameters
        ----------
        df : pd.DataFrame
            Input features to validate.
        allow_extra_columns : bool, default True
            If True, ignore and drop extra columns not in the schema.
            If False, raise ValueError when extra columns are present.

        Returns
        -------
        pd.DataFrame
            Aligned DataFrame containing only the expected features in exact order.

        Raises
        ------
        ValueError
            If duplicate columns exist, required features are missing, or non-numeric
            types appear in expected numeric columns.
        """
        # Normalize column names: strip surrounding whitespace
        norm_cols = [str(c).strip() for c in df.columns]

        # Check duplicate column names after whitespace normalization
        seen: Dict[str, int] = {}
        for c in norm_cols:
            seen[c] = seen.get(c, 0) + 1
        duplicates = [col for col, count in seen.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate column names detected after normalization: {duplicates}"
            )

        # Work on a view/copy with normalized column names
        df = df.copy(deep=False)
        df.columns = norm_cols

        # Check for missing required features
        df_cols_set = set(df.columns)
        missing_features = [f for f in self.feature_names if f not in df_cols_set]
        if missing_features:
            raise ValueError(
                f"Input DataFrame is missing required features: {missing_features}"
            )

        # Check for target column presence in inference feature input (leakage prevention)
        if self.target_column in df.columns:
            df = df.drop(columns=[self.target_column])

        if not allow_extra_columns:
            extra = [c for c in df.columns if c not in self.feature_names]
            if extra:
                raise ValueError(
                    f"Unexpected additional columns present in feature input: {extra}"
                )

        # Reorder columns to exactly match schema
        aligned = df[self.feature_names].copy()

        # Validate that expected numeric features are indeed numeric
        for col in self.numeric_features:
            if not np.issubdtype(aligned[col].dtype, np.number):
                raise TypeError(
                    f"Feature '{col}' is expected to be numeric, but has dtype '{aligned[col].dtype}'. "
                    f"Arbitrary string coercion is disallowed."
                )

        return aligned

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary representation."""
        return {
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "metadata_excluded": self.metadata_excluded,
            "target_column": self.target_column,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Reconstruct schema from dictionary."""
        return cls(
            feature_names=data["feature_names"],
            numeric_features=data["numeric_features"],
            metadata_excluded=data.get("metadata_excluded", []),
            target_column=data.get("target_column", "Label"),
        )


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_column: Optional[str] = "Label",
    feature_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    drop_metadata: bool = True,
    drop_non_numeric: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], FeatureSchema]:
    """
    Prepare and validate feature matrix X and optional target series y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and optionally the target column.
    target_column : Optional[str], default 'Label'
        Name of target column. If present in df, it will be extracted into y.
    feature_columns : Optional[List[str]], default None
        Explicit list of expected feature columns. If provided, checks that all
        are present and preserves their exact ordering.
    exclude_columns : Optional[List[str]], default None
        List of additional column names to explicitly exclude from features.
    drop_metadata : bool, default True
        If True, automatically exclude standard metadata identifiers
        (e.g., 'Flow ID', 'Source IP', 'Timestamp').
    drop_non_numeric : bool, default True
        If True, drop non-numeric columns from features and record them.
        If False, raise TypeError if non-numeric feature columns remain.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series], FeatureSchema]
        (X, y, schema) where X has target and metadata excluded, preserving column order.

    Raises
    ------
    KeyError
        If target_column or any specified feature_columns are missing.
    ValueError
        If duplicate columns exist or any feature column is 100% missing.
    TypeError
        If drop_non_numeric is False and non-numeric feature columns remain.
    """
    # 1. Reject duplicate column names
    seen: Dict[str, int] = {}
    for c in df.columns:
        seen[str(c)] = seen.get(str(c), 0) + 1
    duplicates = [col for col, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate column names detected in DataFrame: {duplicates}")

    df_work = df.copy()

    # 2. Extract target if target_column is specified and present
    y: Optional[pd.Series] = None
    if target_column is not None:
        if target_column in df_work.columns:
            y = df_work[target_column].copy()
            df_work = df_work.drop(columns=[target_column])
        elif feature_columns is None:
            raise KeyError(
                f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}"
            )

    # 3. Handle metadata and explicitly excluded columns
    columns_to_exclude: Set[str] = set()
    if drop_metadata:
        columns_to_exclude.update(
            [col for col in DEFAULT_METADATA_COLUMNS if col in df_work.columns]
        )
    if exclude_columns is not None:
        columns_to_exclude.update(
            [col for col in exclude_columns if col in df_work.columns]
        )

    excluded_list = sorted(list(columns_to_exclude))
    if columns_to_exclude:
        df_work = df_work.drop(columns=list(columns_to_exclude))

    # 4. If an explicit feature schema was supplied, validate and align order
    if feature_columns is not None:
        missing = [f for f in feature_columns if f not in df_work.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")
        X = df_work[feature_columns].copy()
    else:
        X = df_work

    # 5. Check non-numeric columns
    non_numeric_cols = [
        col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)
    ]
    if non_numeric_cols:
        if drop_non_numeric:
            X = X.drop(columns=non_numeric_cols)
        else:
            raise TypeError(
                f"Non-numeric columns detected in feature set: {non_numeric_cols}. "
                f"Silent coercion to numeric is disallowed."
            )

    # 6. Reject any column that is 100% missing values
    if len(X) > 0:
        all_missing_cols = [col for col in X.columns if X[col].isna().all()]
        if all_missing_cols:
            raise ValueError(
                f"Feature column(s) contain 100% missing values: {all_missing_cols}. "
                f"Cannot train or evaluate estimators on completely missing features."
            )

    # 7. Verify target is strictly excluded from X (leakage assertion)
    if target_column is not None:
        assert target_column not in X.columns, f"Target '{target_column}' leaked into features"

    # Build schema
    numeric_cols = list(X.columns)
    schema = FeatureSchema(
        feature_names=list(X.columns),
        numeric_features=numeric_cols,
        metadata_excluded=excluded_list,
        target_column=target_column or "Label",
    )

    return X, y, schema


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: str = "Label",
    reject_missing: bool = True,
    allowed_labels: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix from target label series (Milestone 2 compatible API)."""
    return split_features_and_target(
        df=df,
        target_column=target_column,
        reject_missing=reject_missing,
        allowed_labels=allowed_labels,
    )
