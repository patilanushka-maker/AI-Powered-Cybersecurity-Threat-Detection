"""
Data Loader Module for AI-Powered Cybersecurity Threat Detection System.

Responsible for discovering, reading, and performing initial schema validation
on raw network flow CSV datasets without hardcoding specific dataset schemas.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Iterator, Optional
import pandas as pd


def normalize_column_names(columns: Any) -> List[str]:
    """
    Convert column names to strings, strip leading and trailing whitespace,
    and verify that no duplicate column names exist after normalization.

    Parameters
    ----------
    columns : Iterable
        Iterable containing column names.

    Returns
    -------
    List[str]
        List of normalized column names.

    Raises
    ------
    ValueError
        If duplicate column names are detected after whitespace stripping.
    """
    normalized = [str(col).strip() for col in columns]
    seen: Dict[str, int] = {}
    for col in normalized:
        seen[col] = seen.get(col, 0) + 1

    duplicates = [col for col, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate column names detected after normalization: {duplicates}"
        )
    return normalized


def load_dataset(
    file_path: Union[str, Path, List[Union[str, Path]]],
    provenance_col: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load network flow records from a CSV file, directory of CSV files, or list of files.

    Parameters
    ----------
    file_path : Union[str, Path, List[Union[str, Path]]]
        Path to a single network flow CSV file, directory containing CSV files,
        comma-separated string of paths, or list of paths.
    provenance_col : Optional[str], default None
        Optional column name to record the source file basename for tracking provenance.
    **kwargs : Any
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded network flow DataFrame with stripped and validated column headers.

    Raises
    ------
    FileNotFoundError
        If the targeted file or directory does not exist.
    ValueError
        If the path is empty, malformed, contains no usable CSV files,
        has mismatched schemas across files, or contains duplicate columns.
    """
    # Handle list or tuple of paths
    if isinstance(file_path, (list, tuple)):
        if not file_path:
            raise ValueError("Empty list of dataset paths provided.")
        dfs = []
        expected_cols = None
        for p in file_path:
            sub_df = load_dataset(p, provenance_col=provenance_col, **kwargs)
            cols_to_check = [c for c in sub_df.columns if c != provenance_col]
            if expected_cols is None:
                expected_cols = cols_to_check
            else:
                if set(cols_to_check) != set(expected_cols):
                    raise ValueError(
                        f"Schema mismatch across datasets: '{Path(p).name}' has differing columns."
                    )
                aligned_order = expected_cols + ([provenance_col] if provenance_col in sub_df.columns else [])
                sub_df = sub_df[aligned_order]
            dfs.append(sub_df)
        return pd.concat(dfs, ignore_index=True)

    # Handle comma-separated string of paths
    if isinstance(file_path, str) and "," in file_path:
        parts = [p.strip() for p in file_path.split(",") if p.strip()]
        return load_dataset(parts, provenance_col=provenance_col, **kwargs)

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    # Handle directory containing multiple CSV files
    if path.is_dir():
        csv_files = list_raw_datasets(path)
        if not csv_files:
            raise ValueError(f"Directory contains no CSV files: {path}")
        return load_dataset(csv_files, provenance_col=provenance_col, **kwargs)

    if not path.is_file():
        raise ValueError(f"Target path is not a regular file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty (0 bytes): {path}")

    try:
        df = pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file contains no data or headers: {path}")

    if df.columns.empty or len(df.columns) == 0:
        raise ValueError(f"CSV file contains no usable columns: {path}")

    # Normalize column names and enforce uniqueness
    df.columns = normalize_column_names(df.columns)

    if provenance_col:
        df[provenance_col] = path.name

    return df


def load_dataset_chunks(
    file_path: Union[str, Path], chunksize: int = 10000, **kwargs: Any
) -> Iterator[pd.DataFrame]:
    """
    Read a CSV file in chunks, yielding DataFrames with normalized column names.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the network flow CSV file.
    chunksize : int, default 10000
        Number of rows per chunk.
    **kwargs : Any
        Additional keyword arguments passed to pandas.read_csv.

    Yields
    ------
    pd.DataFrame
        Normalized DataFrame chunk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Target path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty (0 bytes): {path}")

    try:
        reader = pd.read_csv(path, chunksize=chunksize, **kwargs)
        first_chunk = True
        normalized_cols: List[str] = []

        for chunk in reader:
            if first_chunk:
                if chunk.columns.empty or len(chunk.columns) == 0:
                    raise ValueError(f"CSV file contains no usable columns: {path}")
                normalized_cols = normalize_column_names(chunk.columns)
                first_chunk = False
            chunk.columns = normalized_cols
            yield chunk
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file contains no data or headers: {path}")


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate an operational summary report of a loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary reporting row count, column count, column names,
        missing-value counts per column, total missing values, and duplicate rows.
    """
    missing_per_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values_per_column": missing_per_col,
        "total_missing_values": total_missing,
        "duplicate_rows_count": duplicate_rows,
    }


def list_raw_datasets(data_dir: Union[str, Path]) -> List[Path]:
    """
    Scan the raw data directory for available network flow CSV files.

    Parameters
    ----------
    data_dir : Union[str, Path]
        Path to the data/raw directory.

    Returns
    -------
    List[Path]
        List of paths to CSV files found in the directory.
    """
    path = Path(data_dir)
    if not path.exists():
        return []
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    return sorted(list(path.glob("*.csv")))
