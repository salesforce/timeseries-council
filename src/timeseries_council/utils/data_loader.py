# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Data Loading Utilities - Unified CSV loading with chronological sorting.
"""

import pandas as pd
from typing import Optional, List, Union
from ..logging import get_logger

logger = get_logger(__name__)


def load_timeseries_csv(
    csv_path: str,
    parse_dates: bool = True,
    sort: bool = True,
    date_col: int = 0,
    index_col: Optional[int] = 0
) -> pd.DataFrame:
    """
    Load a time series CSV file with optional date parsing and chronological sorting.
    
    This is the canonical way to load time series data in the application.
    All tools and the orchestrator should use this function to ensure consistency.
    
    Args:
        csv_path: Path to the CSV file
        parse_dates: Whether to parse the first column as dates
        sort: Whether to sort by index (chronologically)
        date_col: Column index to parse as dates (default: 0, first column)
        index_col: Column to use as index (default: 0, first column; use None for no index)
        
    Returns:
        DataFrame with dates parsed and sorted chronologically
    """
    try:
        # Load without index first to inspect columns
        df = pd.read_csv(csv_path)
        
        # If user explicitly requested column 0 but it's an integer sequence (like Unnamed: 0),
        # try to find a real date column instead or fallback to None
        if index_col == 0 and len(df.columns) > 1:
            first_col_name = df.columns[0]
            if str(first_col_name).startswith("Unnamed") or df[first_col_name].dtype.kind in 'iufc':
                # Try to find a date/time column
                found_date_col = None
                for col in df.columns:
                    if col != first_col_name:
                        # Check name heuristics
                        col_lower = str(col).lower()
                        if any(k in col_lower for k in ['time', 'date', 'timestamp', 'ds']):
                            found_date_col = col
                            break
                        # Check data heuristics
                        if df[col].dtype == object:
                            try:
                                pd.to_datetime(df[col].head(5))
                                found_date_col = col
                                break
                            except:
                                pass
                
                if found_date_col:
                    index_col = found_date_col
                    date_col = found_date_col
                elif str(first_col_name).startswith("Unnamed"):
                    index_col = None # don't use the Unnamed index as index

        if index_col is not None:
            if isinstance(index_col, int) and index_col < len(df.columns):
                index_col = df.columns[index_col]
            if parse_dates and index_col:
                df[index_col] = pd.to_datetime(df[index_col], errors='coerce')
            df.set_index(index_col, inplace=True)
            
        if sort and df.index is not None and len(df) > 0:
            try:
                df = df.sort_index()
            except Exception:
                pass
            logger.debug(f"Loaded and sorted {csv_path}: {len(df)} rows")
        else:
            logger.debug(f"Loaded {csv_path}: {len(df)} rows")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        raise


def get_date_range(df: pd.DataFrame) -> tuple:
    """
    Get the date range of a DataFrame with DatetimeIndex.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if df.index is None or len(df) == 0:
        return (None, None)
    
    return (str(df.index[0]), str(df.index[-1]))


def infer_frequency(series: pd.Series) -> str:
    """
    Infer the frequency of a time series, with fallback for irregular data.
    
    When pd.infer_freq() fails (returns None) for data with gaps or irregular
    sampling, this function calculates the median time difference to determine
    the actual frequency.
    
    Args:
        series: A pandas Series with DatetimeIndex
        
    Returns:
        Frequency string (e.g., 'T' for minute, 'H' for hourly, 'D' for daily)
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return 'D'
    
    if len(series) < 2:
        return 'D'
    
    # Try pandas infer_freq first
    freq = pd.infer_freq(series.index)
    if freq is not None:
        return freq
    
    # Calculate median time difference
    try:
        time_diffs = pd.Series(series.index).diff().dropna()
        if len(time_diffs) == 0:
            return 'D'
        
        median_diff = time_diffs.median()
        
        if not hasattr(median_diff, 'total_seconds'):
            return 'D'
        
        seconds = median_diff.total_seconds()
        
        # Map seconds to frequency string
        if seconds <= 1:
            return 'S'  # Second
        elif seconds <= 60:
            return 'T'  # Minute (T is legacy, 'min' also works)
        elif seconds <= 3600:
            # Calculate how many minutes
            minutes = int(round(seconds / 60))
            return f'{minutes}T' if minutes > 1 else 'T'
        elif seconds <= 86400:
            # Calculate how many hours
            hours = int(round(seconds / 3600))
            return f'{hours}H' if hours > 1 else 'H'
        elif seconds <= 604800:
            # Calculate how many days
            days = int(round(seconds / 86400))
            return f'{days}D' if days > 1 else 'D'
        else:
            # Weekly or longer
            weeks = int(round(seconds / 604800))
            if weeks <= 4:
                return f'{weeks}W' if weeks > 1 else 'W'
            else:
                return 'M'  # Monthly approximation
                
    except Exception as e:
        logger.warning(f"Could not infer frequency: {e}")
        return 'D'

