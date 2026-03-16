# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Shared utilities for tool functions.
"""

import pandas as pd
from typing import Optional


def prepare_series(
    csv_path: Optional[str] = None,
    target_col: Optional[str] = None,
    series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Load and prepare a time series from either a csv_path or a provided Series.

    Args:
        csv_path: Path to CSV file (used with target_col).
        target_col: Column name to extract from CSV.
        series: Pre-loaded pd.Series with DatetimeIndex.

    Returns:
        Cleaned, sorted pd.Series with DatetimeIndex.

    Raises:
        ValueError: If neither series nor csv_path+target_col are provided,
                    or if target_col is missing from the CSV.
    """
    if series is not None:
        s = series.copy()
    elif csv_path is not None and target_col is not None:
        from ..utils import load_timeseries_csv
        df = load_timeseries_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found. Available: {list(df.columns)}"
            )
        s = df[target_col].dropna()
    else:
        raise ValueError(
            "Either 'series' (pd.Series with DatetimeIndex) or both "
            "'csv_path' and 'target_col' must be provided"
        )

    s = pd.to_numeric(s, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    return s.sort_index()


def prepare_dataframe(
    csv_path: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load and prepare a DataFrame from either a csv_path or a provided DataFrame.

    Args:
        csv_path: Path to CSV file.
        data: Pre-loaded pd.DataFrame with DatetimeIndex.

    Returns:
        DataFrame with DatetimeIndex, sorted by index.

    Raises:
        ValueError: If neither data nor csv_path are provided.
    """
    if data is not None:
        df = data.copy()
    elif csv_path is not None:
        from ..utils import load_timeseries_csv
        df = load_timeseries_csv(csv_path)
    else:
        raise ValueError(
            "Either 'data' (pd.DataFrame with DatetimeIndex) or "
            "'csv_path' must be provided"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()
