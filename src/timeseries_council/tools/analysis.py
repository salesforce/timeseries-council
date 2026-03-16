# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Analysis tools for the orchestrator.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)


def describe_series(
    csv_path: str = None,
    target_col: str = None,
    window: Optional[int] = None,
    series: pd.Series = None,
) -> Dict[str, Any]:
    """
    Describe a time series with basic statistics.

    Args:
        csv_path: Path to CSV file
        target_col: Column to analyze
        window: Optional window of recent points to focus on
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)

    Returns:
        Dict with statistical summary
    """
    logger.info(f"Describing series: {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if window:
            series = series.tail(window)

        # Basic stats
        stats = {
            "success": True,
            "count": int(len(series)),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "latest_value": round(float(series.iloc[-1]), 2),
            "start_date": str(series.index[0]),
            "end_date": str(series.index[-1]),
        }

        # Trend detection
        if len(series) >= 3:
            x = np.arange(len(series))
            slope = np.polyfit(x, series.values, 1)[0]
            if slope > 0.01 * stats["std"]:
                stats["trend"] = "increasing"
            elif slope < -0.01 * stats["std"]:
                stats["trend"] = "decreasing"
            else:
                stats["trend"] = "stable"
            stats["trend_slope"] = round(float(slope), 4)
            
            # Calculate trend data for plotting (using rolling mean)
            # Window size: ~7% of data length, minimum 3, maximum 30
            window_size = max(3, min(30, len(series) // 15))
            trend_values = series.rolling(window=window_size, center=True, min_periods=1).mean()
            
            # Prepare trend data for plotting (drop NaN values)
            trend_series = trend_values.dropna()
            stats["trend_data"] = {
                "timestamps": [str(t) for t in trend_series.index.tolist()],
                "values": [round(float(v), 2) for v in trend_series.values.tolist()]
            }
        else:
            stats["trend"] = "insufficient_data"

        # Total change
        if len(series) >= 2:
            pct_change = ((series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0])) * 100
            stats["total_change_pct"] = round(float(pct_change), 2)

        logger.info(f"Described series: {stats['count']} points, trend={stats['trend']}")
        return stats

    except Exception as e:
        logger.error(f"Describe error: {e}")
        return {"success": False, "error": str(e)}


def decompose_series(
    csv_path: str = None,
    target_col: str = None,
    period: int = 7,
    model: str = 'additive',
    series: pd.Series = None,
) -> Dict[str, Any]:
    """
    Decompose time series into trend, seasonal, and residual components.

    Args:
        csv_path: Path to CSV file
        target_col: Column to decompose
        period: Seasonal period (default 7 for weekly)
        model: "additive" (default) or "multiplicative"
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)

    Returns:
        Dict with decomposition results
    """
    logger.info(f"Decomposing series: {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 2 * period:
            return {
                "success": False,
                "error": f"Need at least {2 * period} data points for period={period}"
            }

        # Perform decomposition
        result = seasonal_decompose(series, model=model, period=period)

        # Extract insights
        trend_values = result.trend.dropna()
        seasonal_values = result.seasonal
        residual_values = result.resid.dropna()
        
        trend_direction = "increasing" if trend_values.iloc[-1] > trend_values.iloc[0] else "decreasing"
        seasonal_range = float(result.seasonal.max() - result.seasonal.min())
        residual_std = float(residual_values.std())

        # Build chart_series - a generic format for plottable data
        chart_series = {
            "trend": {
                "name": "Trend",
                "timestamps": [str(t) for t in trend_values.index.tolist()],
                "values": [round(float(v), 2) for v in trend_values.values.tolist()],
                "color": "#2563eb",
                "description": f"Smoothed trend component ({trend_direction})"
            },
            "seasonal": {
                "name": "Seasonal Pattern",
                "timestamps": [str(t) for t in seasonal_values.index.tolist()],
                "values": [round(float(v), 2) for v in seasonal_values.values.tolist()],
                "color": "#16a34a",
                "description": f"Seasonal component (period={period})"
            },
            "residual": {
                "name": "Residual (Noise)",
                "timestamps": [str(t) for t in residual_values.index.tolist()],
                "values": [round(float(v), 2) for v in residual_values.values.tolist()],
                "color": "#dc2626",
                "description": "Random noise/residual component"
            }
        }

        return {
            "success": True,
            "period": period,
            "trend_direction": trend_direction,
            "trend_start": round(float(trend_values.iloc[0]), 2),
            "trend_end": round(float(trend_values.iloc[-1]), 2),
            "trend_change_pct": round(((trend_values.iloc[-1] - trend_values.iloc[0]) / abs(trend_values.iloc[0])) * 100, 2),
            "seasonal_amplitude": round(seasonal_range, 2),
            "seasonal_pattern": [round(float(result.seasonal.iloc[i]), 2) for i in range(min(period, len(result.seasonal)))],
            "residual_std": round(residual_std, 2),
            "noise_ratio": round(residual_std / series.std() * 100, 2),
            # Generic plottable data
            "chart_series": chart_series,
        }

    except ImportError:
        logger.error("statsmodels not installed")
        return {
            "success": False,
            "error": "statsmodels not installed. Run: pip install statsmodels"
        }
    except Exception as e:
        logger.error(f"Decompose error: {e}")
        return {"success": False, "error": str(e)}


def compare_series(
    csv_path: str = None,
    columns: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    data: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Compare multiple columns - correlations and relative statistics.

    Args:
        csv_path: Path to CSV file
        columns: List of columns to compare (default: all numeric columns)
        target_col: Ignored - for orchestrator compatibility
        data: Pre-loaded pd.DataFrame with DatetimeIndex (alternative to csv_path)

    Returns:
        Dict with correlation matrix and comparative stats
    """
    logger.info(f"Comparing series: {csv_path or 'dataframe'}")

    try:
        from ._utils import prepare_dataframe
        df = prepare_dataframe(csv_path=csv_path, data=data)

        # Select columns
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return {
                    "success": False,
                    "error": f"Columns not found: {missing}. Available: {list(df.columns)}"
                }
            df = df[columns]
        else:
            df = df.select_dtypes(include=[np.number])

        if len(df.columns) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 numeric columns. Found: {list(df.columns)}"
            }

        # Compute correlation matrix
        corr_matrix = df.corr().round(3)

        # Find strongest correlations
        correlations = []
        cols = corr_matrix.columns.tolist()
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                correlations.append({
                    "pair": f"{col1} vs {col2}",
                    "correlation": float(corr_matrix.loc[col1, col2])
                })

        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Comparative statistics
        stats = {}
        for col in df.columns:
            series = df[col].dropna()
            stats[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }

        logger.info(f"Compared {len(df.columns)} columns")

        return {
            "success": True,
            "columns_compared": list(df.columns),
            "correlation_matrix": corr_matrix.to_dict(),
            "top_correlations": correlations[:5],
            "column_stats": stats
        }

    except Exception as e:
        logger.error(f"Compare error: {e}")
        return {"success": False, "error": str(e)}


def compare_periods(
    csv_path: str = None,
    target_col: str = None,
    periods: List[str] = None,
    period1: str = None,
    period2: str = None,
    groups: int = None,
    group_names: List[str] = None,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare statistics across different time periods (months, quarters, etc.).

    Args:
        csv_path: Path to CSV file
        target_col: Column to analyze
        periods: List of periods to compare (e.g., ["March", "April", "May"] or ["q1", "q3"])
        period1: First period (alternative to periods list)
        period2: Second period (alternative to periods list)
        groups: Number of groups to aggregate periods into (e.g., 2 for two quarters from 6 months)
        group_names: Optional names for groups (e.g., ["Q3", "Q4"])
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)
        **kwargs: Additional arguments

    Returns:
        Dict with comparative statistics for each period or group
    """
    logger.info(f"Comparing periods on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series
        from ..utils.date_parsing import (
            parse_month_reference,
            filter_series_by_month,
            parse_month_name,
            get_available_months,
            MONTH_NUMBER_TO_NAME
        )

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        # Build list of periods to compare
        periods_to_compare = []
        if periods:
            periods_to_compare = periods
        elif period1 and period2:
            periods_to_compare = [period1, period2]
        elif period1:
            periods_to_compare = [period1]
        else:
            # Default: compare all available months
            available = get_available_months(series)
            if len(available) > 12:
                available = available[-12:]  # Last 12 months
            periods_to_compare = [f"{m['month_name']} {m['year']}" for m in available]

        if len(periods_to_compare) < 1:
            return {
                "success": False,
                "error": "No periods specified to compare",
                "available_months": get_available_months(series)
            }

        # Detect if the LLM sent grouped periods (nested lists)
        # e.g., [["January", "February", "March"], ["July", "August", "September"]]
        # vs flat periods ["January", "February", "March"]
        def detect_period_groups(period_list):
            """
            Detect if period_list contains grouped periods (nested lists).
            
            Returns:
                tuple: (is_grouped, groups_or_flat_list)
                - If grouped: (True, [[months_group1], [months_group2], ...])
                - If flat: (False, [month1, month2, ...])
            """
            has_nested_lists = any(isinstance(item, list) for item in period_list)
            
            if has_nested_lists:
                # Parse as groups - each sub-list is a group to aggregate
                groups = []
                for item in period_list:
                    if isinstance(item, list):
                        # Flatten any deeply nested structure within the group
                        flat_group = []
                        def flatten(lst):
                            for el in lst:
                                if isinstance(el, list):
                                    flatten(el)
                                else:
                                    flat_group.append(str(el))
                        flatten(item)
                        groups.append(flat_group)
                    else:
                        # Single item treated as its own group
                        groups.append([str(item)])
                return True, groups
            else:
                # Flat list of individual periods
                return False, [str(item) for item in period_list]

        is_grouped, parsed_periods = detect_period_groups(periods_to_compare)
        logger.info(f"Period grouping detected: is_grouped={is_grouped}, periods={parsed_periods}")

        # Analyze periods - handle both grouped and flat cases
        period_stats = []
        all_values = []

        def analyze_single_period(period_spec, series):
            """Analyze a single period (month) and return stats dict or None."""
            month_info = parse_month_reference(str(period_spec), series)
            
            if not month_info.get("success"):
                logger.warning(f"Could not parse period: {period_spec}")
                return None, None

            # If ambiguous, use the most recent year
            if month_info.get("ambiguous"):
                match = sorted(month_info["matches"], key=lambda x: x["year"], reverse=True)[0]
            else:
                match = month_info["matches"][0]

            month_num = month_info["month"]
            year = match["year"]
            
            # Filter to this period
            period_data = filter_series_by_month(series, month_num, year)
            
            if len(period_data) == 0:
                return None, None

            period_name = f"{MONTH_NUMBER_TO_NAME[month_num]} {year}"
            return period_name, period_data

        if is_grouped:
            # Grouped periods: aggregate data across each group of months
            for group_idx, month_group in enumerate(parsed_periods):
                group_data_list = []
                month_names = []
                
                for month_spec in month_group:
                    period_name, period_data = analyze_single_period(month_spec, series)
                    if period_data is not None and len(period_data) > 0:
                        group_data_list.append(period_data)
                        month_names.append(period_name)
                
                if not group_data_list:
                    logger.warning(f"No valid data for group {group_idx}: {month_group}")
                    continue
                
                # Concatenate all data for this group
                group_data = pd.concat(group_data_list).sort_index()
                
                # Create a descriptive group name (e.g., "January 2025 - March 2025" or "Q1")
                if len(month_names) == 3:
                    # Check if it looks like a quarter
                    first_month = month_names[0].split()[0]
                    if first_month in ['January', 'April', 'July', 'October']:
                        quarter_map = {'January': 'Q1', 'April': 'Q2', 'July': 'Q3', 'October': 'Q4'}
                        group_name = f"{quarter_map.get(first_month, 'Group')} {month_names[0].split()[-1]}"
                    else:
                        group_name = f"{month_names[0]} - {month_names[-1]}"
                else:
                    group_name = f"{month_names[0]} - {month_names[-1]}" if len(month_names) > 1 else month_names[0]
                
                stats = {
                    "period": group_name,
                    "months_included": month_names,
                    "count": int(len(group_data)),
                    "mean": round(float(group_data.mean()), 2),
                    "median": round(float(group_data.median()), 2),
                    "std": round(float(group_data.std()), 2),
                    "min": round(float(group_data.min()), 2),
                    "max": round(float(group_data.max()), 2),
                    "sum": round(float(group_data.sum()), 2),
                    "start_date": str(group_data.index[0]),
                    "end_date": str(group_data.index[-1]),
                }
                
                # Trend within group
                if len(group_data) >= 3:
                    x = np.arange(len(group_data))
                    slope = np.polyfit(x, group_data.values, 1)[0]
                    if slope > 0.01 * stats["std"]:
                        stats["trend"] = "increasing"
                    elif slope < -0.01 * stats["std"]:
                        stats["trend"] = "decreasing"
                    else:
                        stats["trend"] = "stable"
                        
                # Change from first to last
                if len(group_data) >= 2:
                    pct_change = ((group_data.iloc[-1] - group_data.iloc[0]) / abs(group_data.iloc[0])) * 100
                    stats["change_pct"] = round(float(pct_change), 2)
                    
                period_stats.append(stats)
                all_values.extend(group_data.tolist())
        else:
            # Flat periods: analyze each period individually
            for period_spec in parsed_periods:
                period_name, period_data = analyze_single_period(period_spec, series)
                
                if period_data is None or len(period_data) == 0:
                    continue

                # Get month number from the period_name
                month_name_only = period_name.split()[0]
                month_num = parse_month_name(month_name_only)
                year = int(period_name.split()[-1])
                
                stats = {
                    "period": period_name,
                    "month": month_num,
                    "year": year,
                    "count": int(len(period_data)),
                    "mean": round(float(period_data.mean()), 2),
                    "median": round(float(period_data.median()), 2),
                    "std": round(float(period_data.std()), 2),
                    "min": round(float(period_data.min()), 2),
                    "max": round(float(period_data.max()), 2),
                    "sum": round(float(period_data.sum()), 2),
                    "start_date": str(period_data.index[0]),
                    "end_date": str(period_data.index[-1]),
                }
                
                # Trend within period
                if len(period_data) >= 3:
                    x = np.arange(len(period_data))
                    slope = np.polyfit(x, period_data.values, 1)[0]
                    if slope > 0.01 * stats["std"]:
                        stats["trend"] = "increasing"
                    elif slope < -0.01 * stats["std"]:
                        stats["trend"] = "decreasing"
                    else:
                        stats["trend"] = "stable"
                        
                # Change from first to last
                if len(period_data) >= 2:
                    pct_change = ((period_data.iloc[-1] - period_data.iloc[0]) / abs(period_data.iloc[0])) * 100
                    stats["change_pct"] = round(float(pct_change), 2)
                    
                period_stats.append(stats)
                all_values.extend(period_data.tolist())

        if len(period_stats) == 0:
            return {
                "success": False,
                "error": "No valid data found for specified periods",
                "available_months": get_available_months(series)
            }

        # If groups is specified, aggregate period_stats into groups
        if groups and groups > 0 and len(period_stats) > 1:
            grouped_stats = []
            periods_per_group = len(period_stats) // groups
            
            for g in range(groups):
                start_idx = g * periods_per_group
                end_idx = start_idx + periods_per_group if g < groups - 1 else len(period_stats)
                group_periods = period_stats[start_idx:end_idx]
                
                if not group_periods:
                    continue
                
                # Aggregate stats for this group
                group_values = []
                for p in group_periods:
                    # We need to re-get the values, or estimate from stats
                    # Using weighted average based on count
                    pass
                
                # Calculate aggregated stats
                total_count = sum(p["count"] for p in group_periods)
                weighted_mean = sum(p["mean"] * p["count"] for p in group_periods) / total_count if total_count > 0 else 0
                
                # Determine group name
                if group_names and g < len(group_names):
                    gname = group_names[g]
                else:
                    first_period = group_periods[0]["period"]
                    last_period = group_periods[-1]["period"]
                    gname = f"{first_period} - {last_period}" if first_period != last_period else first_period
                
                grouped_stats.append({
                    "period": gname,
                    "periods_included": [p["period"] for p in group_periods],
                    "count": total_count,
                    "mean": round(weighted_mean, 2),
                    "median": round(float(np.median([p["median"] for p in group_periods])), 2),
                    "min": round(min(p["min"] for p in group_periods), 2),
                    "max": round(max(p["max"] for p in group_periods), 2),
                    "sum": round(sum(p["sum"] for p in group_periods), 2),
                    "start_date": group_periods[0]["start_date"],
                    "end_date": group_periods[-1]["end_date"],
                })
            
            # Replace period_stats with grouped stats
            period_stats = grouped_stats

        # Calculate comparison insights
        comparison = {}
        if len(period_stats) >= 2:
            means = [p["mean"] for p in period_stats]
            best_period = period_stats[means.index(max(means))]
            worst_period = period_stats[means.index(min(means))]
            
            comparison = {
                "highest_mean": {
                    "period": best_period["period"],
                    "value": best_period["mean"]
                },
                "lowest_mean": {
                    "period": worst_period["period"],
                    "value": worst_period["mean"]
                },
                "mean_difference": round(max(means) - min(means), 2),
                "mean_difference_pct": round((max(means) - min(means)) / min(means) * 100, 2) if min(means) != 0 else 0,
            }
            
            # Period-over-period change
            if len(period_stats) == 2:
                p1, p2 = period_stats[0], period_stats[1]
                comparison["change"] = {
                    "from": p1["period"],
                    "to": p2["period"],
                    "mean_change": round(p2["mean"] - p1["mean"], 2),
                    "mean_change_pct": round((p2["mean"] - p1["mean"]) / abs(p1["mean"]) * 100, 2) if p1["mean"] != 0 else 0,
                }

        return {
            "success": True,
            "periods_compared": len(period_stats),
            "period_stats": period_stats,
            "comparison": comparison,
            "overall": {
                "total_points": len(all_values),
                "overall_mean": round(float(np.mean(all_values)), 2),
                "overall_std": round(float(np.std(all_values)), 2),
            }
        }

    except Exception as e:
        logger.error(f"Compare periods error: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


# Tool registration info
DESCRIBE_TOOL = {
    "name": "describe_series",
    "function": describe_series,
    "description": "Get statistical summary and trend analysis of a time series",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to analyze",
        "window": "Optional: analyze only last N points"
    }
}

DECOMPOSE_TOOL = {
    "name": "decompose_series",
    "function": decompose_series,
    "description": "Decompose time series into trend, seasonal, and residual components",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to decompose",
        "period": "Seasonal period (default 7 for weekly patterns)"
    }
}

COMPARE_TOOL = {
    "name": "compare_series",
    "function": compare_series,
    "description": "Compare multiple columns - correlations and relative statistics",
    "parameters": {
        "csv_path": "Path to CSV file",
        "columns": "Optional list of columns to compare (default: all numeric)"
    }
}

COMPARE_PERIODS_TOOL = {
    "name": "compare_periods",
    "function": compare_periods,
    "description": "Compare statistics across time periods (months). Use for 'compare March and April', 'which month had highest/lowest'",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to analyze",
        "periods": "List of periods to compare (e.g., ['March 2024', 'April 2024'])",
        "period1": "First period (alternative to periods list)",
        "period2": "Second period (alternative to periods list)"
    }
}
