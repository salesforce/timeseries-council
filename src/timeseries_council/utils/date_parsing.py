# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Date parsing utilities for month-based NLP recognition.
Handles parsing of month references from natural language like "September", "March 2024", etc.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import re
from datetime import datetime


# Month name to number mapping
MONTH_NAMES = {
    'january': 1, 'jan': 1, '1': 1,
    'february': 2, 'feb': 2, '2': 2,
    'march': 3, 'mar': 3, '3': 3,
    'april': 4, 'apr': 4, '4': 4,
    'may': 5, '5': 5,
    'june': 6, 'jun': 6, '6': 6,
    'july': 7, 'jul': 7, '7': 7,
    'august': 8, 'aug': 8, '8': 8,
    'september': 9, 'sep': 9, 'sept': 9, '9': 9,
    'october': 10, 'oct': 10, '10': 10,
    'november': 11, 'nov': 11, '11': 11,
    'december': 12, 'dec': 12, '12': 12,
}

MONTH_NUMBER_TO_NAME = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


def parse_month_name(month_str: str) -> Optional[int]:
    """
    Parse a month name/abbreviation to month number (1-12).
    
    Args:
        month_str: Month name like "September", "Sep", "9"
    
    Returns:
        Month number (1-12) or None if not recognized
    """
    if month_str is None:
        return None
    clean = month_str.lower().strip()
    return MONTH_NAMES.get(clean)


def parse_month_reference(spec: str, series: pd.Series) -> Dict[str, Any]:
    """
    Parse a month reference and find matching data ranges in the series.
    
    Args:
        spec: Month specification like "September", "March 2024", "Sep 24"
        series: Time series with DatetimeIndex
    
    Returns:
        {
            "success": True/False,
            "month": 9,  # 1-12
            "year": 2024 or None,  # if specified
            "matches": [{"year": 2024, "start_idx": 100, "end_idx": 130}, ...],
            "ambiguous": True/False,  # multiple years have this month
            "message": "..."  # description or error message
        }
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return {
                "success": False,
                "error": "Series does not have datetime index"
            }
    
    spec_lower = spec.lower().strip()
    
    # Try to extract month and optional year
    month = None
    year = None
    
    # Pattern 1: "September 2024" or "Sep 2024"
    match = re.match(r'([a-zA-Z]+)\s*(\d{4})', spec_lower)
    if match:
        month = parse_month_name(match.group(1))
        year = int(match.group(2))
    
    # Pattern 2: "September 24" or "Sep 24" (2-digit year)
    if month is None:
        match = re.match(r'([a-zA-Z]+)\s*(\d{2})$', spec_lower)
        if match:
            month = parse_month_name(match.group(1))
            year_2digit = int(match.group(2))
            # Assume 2000s for 2-digit years
            year = 2000 + year_2digit if year_2digit < 50 else 1900 + year_2digit
    
    # Pattern 3: "September" or "Sep" (no year)
    if month is None:
        month = parse_month_name(spec_lower.split()[0] if ' ' in spec_lower else spec_lower)
    
    if month is None:
        return {
            "success": False,
            "error": f"Could not parse month from '{spec}'"
        }
    
    # Find all occurrences of this month in the series
    matches = []
    years_with_month = series.index.to_series().groupby(series.index.year).apply(
        lambda x: x[x.index.month == month]
    )
    
    for yr in series.index.year.unique():
        mask = (series.index.month == month) & (series.index.year == yr)
        if mask.any():
            indices = series.index[mask]
            start_idx = series.index.get_loc(indices[0])
            end_idx = series.index.get_loc(indices[-1])
            
            # Handle slice returns from get_loc
            if isinstance(start_idx, slice):
                start_idx = start_idx.start
            if isinstance(end_idx, slice):
                end_idx = end_idx.stop - 1
            
            matches.append({
                "year": int(yr),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx) + 1,  # exclusive end
                "count": int(mask.sum()),
                "date_range": f"{indices[0].strftime('%Y-%m-%d')} to {indices[-1].strftime('%Y-%m-%d')}"
            })
    
    if not matches:
        available_months = get_available_months(series)
        return {
            "success": False,
            "error": f"No data found for {MONTH_NUMBER_TO_NAME[month]}",
            "available_months": available_months
        }
    
    # If year was specified, filter to that year
    if year is not None:
        matches = [m for m in matches if m["year"] == year]
        if not matches:
            years_available = [m["year"] for m in matches] if matches else []
            return {
                "success": False,
                "error": f"No data for {MONTH_NUMBER_TO_NAME[month]} {year}",
                "available_years": years_available
            }
    
    ambiguous = len(matches) > 1 and year is None
    
    return {
        "success": True,
        "month": month,
        "month_name": MONTH_NUMBER_TO_NAME[month],
        "year": year,
        "matches": matches,
        "ambiguous": ambiguous,
        "message": f"Found {MONTH_NUMBER_TO_NAME[month]} in {len(matches)} year(s)" if ambiguous else f"Found {MONTH_NUMBER_TO_NAME[month]} {matches[0]['year']}"
    }


def get_available_months(series: pd.Series) -> List[Dict[str, Any]]:
    """Get list of available months in the series."""
    if not isinstance(series.index, pd.DatetimeIndex):
        return []
    
    available = []
    for yr in sorted(series.index.year.unique()):
        months_in_year = sorted(series.index[series.index.year == yr].month.unique())
        for m in months_in_year:
            available.append({
                "year": int(yr),
                "month": int(m),
                "month_name": MONTH_NUMBER_TO_NAME[m],
                "label": f"{MONTH_NUMBER_TO_NAME[m]} {yr}"
            })
    return available


def filter_series_by_month(
    series: pd.Series, 
    month: int, 
    year: Optional[int] = None
) -> pd.Series:
    """
    Filter series to only include data from specified month/year.
    
    Args:
        series: Time series with DatetimeIndex
        month: Month number (1-12)
        year: Optional year (if None, includes all years with that month)
    
    Returns:
        Filtered series
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    mask = series.index.month == month
    if year is not None:
        mask = mask & (series.index.year == year)
    
    return series[mask]


def get_context_before_month(
    series: pd.Series, 
    month: int, 
    year: int
) -> pd.Series:
    """
    Get all data before the specified month (for forecasting context).
    
    Args:
        series: Time series with DatetimeIndex
        month: Target month (1-12)
        year: Target year
    
    Returns:
        Series containing all data before the target month
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    # Find the first day of the target month
    target_start = pd.Timestamp(year=year, month=month, day=1)
    
    # Return all data before that date
    return series[series.index < target_start]


def get_month_data_with_context(
    series: pd.Series,
    month: int,
    year: int,
    context_months: int = 0
) -> Tuple[pd.Series, pd.Series]:
    """
    Get target month data and optional preceding context.
    
    Args:
        series: Time series with DatetimeIndex
        month: Target month (1-12)
        year: Target year
        context_months: Number of preceding months to include as context
    
    Returns:
        Tuple of (context_series, target_series)
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    # Get target month data
    target_mask = (series.index.month == month) & (series.index.year == year)
    target_series = series[target_mask]
    
    if context_months > 0:
        # Calculate context start date
        context_start = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=context_months)
        target_start = pd.Timestamp(year=year, month=month, day=1)
        
        context_mask = (series.index >= context_start) & (series.index < target_start)
        context_series = series[context_mask]
    else:
        context_series = pd.Series(dtype=series.dtype)
    
    return context_series, target_series


def filter_series_by_month_range(
    series: pd.Series,
    start_month: int,
    end_month: int,
    year: Optional[int] = None
) -> pd.Series:
    """
    Filter series to only include data from start_month to end_month (inclusive).
    
    Args:
        series: Time series with DatetimeIndex
        start_month: Start month number (1-12)
        end_month: End month number (1-12), can wrap around (e.g., 10-2 for Oct-Feb)
        year: Optional year (if None, includes all years)
    
    Returns:
        Filtered series
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    months = series.index.month
    
    # Handle month range (supports wrap-around like Oct-Feb)
    if start_month <= end_month:
        # Normal range (e.g., 9-12 for Sep-Dec)
        month_mask = (months >= start_month) & (months <= end_month)
    else:
        # Wrap-around range (e.g., 10-2 for Oct-Feb)
        month_mask = (months >= start_month) | (months <= end_month)
    
    if year is not None:
        month_mask = month_mask & (series.index.year == year)
    
    return series[month_mask]


def get_context_before_month_range(
    series: pd.Series,
    start_month: int,
    year: int
) -> pd.Series:
    """
    Get all data before the specified start month (for forecasting context).
    
    Args:
        series: Time series with DatetimeIndex
        start_month: Start month of the target range (1-12)
        year: Target year
    
    Returns:
        Series containing all data before the target start month
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    # Find the first day of the start month
    target_start = pd.Timestamp(year=year, month=start_month, day=1)
    
    # Return all data before that date
    return series[series.index < target_start]


def parse_date_spec(
    spec: str, 
    series: pd.Series, 
    default_year: Optional[int] = None
) -> Optional[pd.Timestamp]:
    """
    Parse a date specification string into a Timestamp.
    
    Supports various formats:
    - "15th Aug", "Aug 15", "15 August"
    - "2024-08-15", "08/15/2024"
    - "Aug 15 2024", "15th Aug 2024"
    
    Args:
        spec: Date specification string
        series: Time series (used to infer year if not specified)
        default_year: Year to use if not specified in spec
    
    Returns:
        pd.Timestamp or None if parsing fails
    """
    if spec is None:
        return None
    
    spec = str(spec).strip()
    
    # Determine default year from series or current
    if default_year is None:
        if isinstance(series.index, pd.DatetimeIndex) and len(series) > 0:
            # Use the most recent year in data
            default_year = series.index.max().year
        else:
            default_year = datetime.now().year
    
    # Clean ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
    spec_clean = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', spec, flags=re.IGNORECASE)
    
    # Try direct parsing first
    try:
        parsed = pd.to_datetime(spec_clean)
        # If year wasn't in the spec, it might default to 1900 or current
        # Check if the year looks reasonable
        if parsed.year < 1970:
            parsed = parsed.replace(year=default_year)
        return parsed
    except:
        pass
    
    # Try with explicit year appended
    try:
        parsed = pd.to_datetime(f"{spec_clean} {default_year}")
        return parsed
    except:
        pass
    
    # Try common formats manually
    formats = [
        "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
        "%d %B", "%d %b", "%B %d", "%b %d",
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    ]
    
    for fmt in formats:
        try:
            if '%Y' in fmt:
                parsed = datetime.strptime(spec_clean, fmt)
            else:
                parsed = datetime.strptime(spec_clean, fmt)
                parsed = parsed.replace(year=default_year)
            return pd.Timestamp(parsed)
        except:
            continue
    
    return None


def parse_date_range(
    start_spec: str,
    end_spec: str,
    series: pd.Series,
    target_year: Optional[int] = None
) -> Dict[str, Any]:
    """
    Parse a date range specification.
    
    Args:
        start_spec: Start date string (e.g., "15th Aug", "Aug 15")
        end_spec: End date string (e.g., "3rd Sept", "Sept 3")
        series: Time series with DatetimeIndex
        target_year: Optional year (if not specified in dates)
    
    Returns:
        {
            "success": True/False,
            "start_date": pd.Timestamp,
            "end_date": pd.Timestamp,
            "error": str (if failed)
        }
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return {
                "success": False,
                "error": "Series does not have datetime index"
            }
    
    # Parse dates
    start_date = parse_date_spec(start_spec, series, target_year)
    end_date = parse_date_spec(end_spec, series, target_year)
    
    if start_date is None:
        return {
            "success": False,
            "error": f"Could not parse start date: '{start_spec}'"
        }
    
    if end_date is None:
        return {
            "success": False,
            "error": f"Could not parse end date: '{end_spec}'"
        }
    
    # Handle year wrap-around (e.g., "Dec 15 to Jan 5")
    if end_date < start_date:
        # Assume end_date is in the next year
        end_date = end_date.replace(year=end_date.year + 1)
    
    # Make end_date inclusive (end of day)
    if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
        end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    return {
        "success": True,
        "start_date": start_date,
        "end_date": end_date,
        "description": f"{start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"
    }


def filter_series_by_date_range(
    series: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.Series:
    """
    Filter series to only include data within the specified date range.
    
    Args:
        series: Time series with DatetimeIndex
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
    
    Returns:
        Filtered series
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    mask = (series.index >= start_date) & (series.index <= end_date)
    return series[mask]


def get_context_before_date(
    series: pd.Series,
    target_date: pd.Timestamp
) -> pd.Series:
    """
    Get all data before the specified date (for forecasting/detection context).
    
    Args:
        series: Time series with DatetimeIndex
        target_date: Target date to get context before
    
    Returns:
        Series containing all data before the target date
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    return series[series.index < target_date]


def get_date_range_with_context(
    series: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> Tuple[pd.Series, pd.Series]:
    """
    Get target date range data and preceding context.
    
    Args:
        series: Time series with DatetimeIndex
        start_date: Start of target range
        end_date: End of target range
    
    Returns:
        Tuple of (context_series, target_series)
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    
    context_series = get_context_before_date(series, start_date)
    target_series = filter_series_by_date_range(series, start_date, end_date)
    
    return context_series, target_series


def build_clarification_response(matches: List[Dict], month_name: str) -> Dict[str, Any]:
    """
    Build a clarification response when month is ambiguous.
    
    Args:
        matches: List of matching year info
        month_name: Name of the month
    
    Returns:
        Response dict with clarification options
    """
    options = []
    for match in sorted(matches, key=lambda x: x["year"]):
        options.append({
            "label": f"{month_name} {match['year']}",
            "value": {"month": MONTH_NAMES[month_name.lower()], "year": match["year"]},
            "data_points": match["count"],
            "date_range": match["date_range"]
        })
    
    return {
        "success": True,
        "needs_clarification": True,
        "message": f"Found {month_name} in {len(matches)} years. Which one do you mean?",
        "options": options
    }
