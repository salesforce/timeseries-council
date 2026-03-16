# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Backtesting tool for the orchestrator.
Test forecasts on historical data with custom context windows and compare with actuals.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from ..logging import get_logger
from ..exceptions import ToolError

logger = get_logger(__name__)


def _to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [_to_python_types(x) for x in obj.tolist()]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, list):
        return [_to_python_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return float('inf')
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    denominator = (np.abs(actual) + np.abs(predicted))
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return float('inf')
    return float(np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_all_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    """Calculate all forecast accuracy metrics."""
    actual_arr = np.array(actual)
    predicted_arr = np.array(predicted)
    
    return {
        "mae": round(calculate_mae(actual_arr, predicted_arr), 4),
        "mape": round(calculate_mape(actual_arr, predicted_arr), 2),
        "smape": round(calculate_smape(actual_arr, predicted_arr), 2),
        "rmse": round(calculate_rmse(actual_arr, predicted_arr), 4),
    }


def parse_window_specification(
    spec: Any,
    series: pd.Series,
    series_length: int,
) -> Tuple[int, str]:
    """
    Parse a window specification into an index.
    
    Args:
        spec: Can be an integer, string like "first 5", "5 months", etc.
        series: The time series (needed to calculate data frequency)
        series_length: Total length of the series
    
    Returns:
        Tuple of (index, description)
    """
    if isinstance(spec, int):
        return min(spec, series_length), f"first {spec} points"
    
    if not isinstance(spec, str):
        # Default: use 80% of data as context
        default_idx = int(series_length * 0.8)
        return default_idx, f"first {default_idx} points (default 80%)"
    
    spec_lower = spec.lower().strip()
    
    # Calculate data frequency (points per time unit)
    points_per_unit = _calculate_points_per_unit(series)
    
    # Time unit keywords and their multipliers to days
    time_units = {
        'year': 365, 'years': 365, 'yr': 365, 'yrs': 365,
        'month': 30, 'months': 30, 'mo': 30,
        'week': 7, 'weeks': 7, 'wk': 7, 'wks': 7,
        'day': 1, 'days': 1, 'd': 1,
    }
    
    # Parse the specification
    is_last = spec_lower.startswith('last')
    is_first = spec_lower.startswith('first')
    
    # Clean prefix
    if is_last:
        spec_clean = spec_lower.replace('last', '').strip()
    elif is_first:
        spec_clean = spec_lower.replace('first', '').strip()
    else:
        spec_clean = spec_lower
    
    # Handle percentages like "half", "80%"
    if 'half' in spec_lower:
        idx = series_length // 2
        if is_last:
            return idx, f"excluding last half"
        return idx, "first half"
    
    if '%' in spec_clean:
        try:
            pct = int(spec_clean.replace('%', '').strip())
            idx = int(series_length * pct / 100)
            if is_last:
                return series_length - idx, f"excluding last {pct}%"
            return idx, f"first {pct}%"
        except ValueError:
            pass
    
    # Try to parse "N units" format (e.g., "5 months", "2 weeks")
    parts = spec_clean.split()
    if len(parts) >= 1:
        try:
            n = int(parts[0])
            
            # Check if there's a time unit
            unit = parts[1] if len(parts) > 1 else None
            
            if unit and unit in time_units:
                # Convert time units to data points
                days = n * time_units[unit]
                points = int(days * points_per_unit.get('day', 1))
                points = max(1, min(points, series_length))
                
                if is_last:
                    return max(0, series_length - points), f"excluding last {n} {unit}"
                return min(points, series_length), f"first {n} {unit} ({points} points)"
            else:
                # No time unit, treat as points
                if is_last:
                    return max(0, series_length - n), f"excluding last {n} points"
                return min(n, series_length), f"first {n} points"
                
        except ValueError:
            pass
    
    # Default: use 80% of data as context
    default_idx = int(series_length * 0.8)
    return default_idx, f"first {default_idx} points (default 80%)"


def _calculate_points_per_unit(series: pd.Series) -> Dict[str, float]:
    """Calculate how many data points correspond to each time unit."""
    result = {'day': 1, 'week': 7, 'month': 30, 'year': 365}
    
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) < 2:
        return result
    
    try:
        # Calculate median time difference between points
        time_diffs = pd.Series(series.index).diff().dropna()
        if len(time_diffs) == 0:
            return result
        
        median_diff = time_diffs.median()
        
        if hasattr(median_diff, 'total_seconds'):
            seconds_per_point = median_diff.total_seconds()
            if seconds_per_point > 0:
                seconds_per_day = 86400
                points_per_day = seconds_per_day / seconds_per_point
                
                result = {
                    'day': points_per_day,
                    'week': points_per_day * 7,
                    'month': points_per_day * 30,
                    'year': points_per_day * 365,
                }
    except Exception:
        pass
    
    return result


def backtest_forecast(
    csv_path: str = None,
    target_col: str = None,
    context_end: Any = None,
    horizon: int = None,
    forecaster: str = "multi",
    model_size: str = "small",
    compare_actual: bool = True,
    target_month: Any = None,
    target_year: int = None,
    start_month: Any = None,
    end_month: Any = None,
    start_date: Any = None,
    end_date: Any = None,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run forecast on historical data with custom context window and compare with actuals.

    Args:
        csv_path: Path to CSV file with time series data
        target_col: Name of the column to forecast
        context_end: Where context window ends. Can be:
            - Integer: index or number of points
            - String: "first 5", "first 5 months", "first half", "80%"
        horizon: Steps to predict after context. If None, predicts remaining data.
        forecaster: Forecaster to use (default: "multi")
        model_size: Model size for applicable forecasters
        compare_actual: Whether to compare with actual values
        target_month: Target month to forecast. Can be:
            - Integer: month number (1-12)
            - String: "September", "Sep", "March 2024"
        target_year: Target year (required if target_month is ambiguous)
        start_month: Start of month range to forecast (for ranges like "Sept to Dec").
            - Integer: month number (1-12)
            - String: "September", "Sep"
        end_month: End of month range to forecast (for ranges like "Sept to Dec").
            - Integer: month number (1-12)
            - String: "December", "Dec"
        start_date: Start date for specific date range (e.g., "Jan 4", "2026-01-04").
            More precise than start_month - use for day-level targeting.
        end_date: End date for specific date range (e.g., "Jan 11", "2026-01-11").
            More precise than end_month - use for day-level targeting.
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)
        **kwargs: Additional forecaster arguments

    Returns:
        Dict with context data, predictions, actuals, and metrics.
        If target_month is ambiguous, returns needs_clarification with options.
    """
    logger.info(f"Backtest forecast on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series
        from ..utils.date_parsing import (
            parse_month_reference,
            parse_month_name,
            get_context_before_month,
            filter_series_by_month,
            filter_series_by_month_range,
            get_context_before_month_range,
            build_clarification_response,
            MONTH_NUMBER_TO_NAME
        )
        from .forecasting import run_forecast

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 5:
            return {
                "success": False,
                "error": "Need at least 5 data points for backtesting"
            }
        
        series_length = len(series)
        
        # Handle specific date range (start_date to end_date) - most precise
        if start_date is not None and end_date is not None:
            try:
                # Parse dates - support various formats
                if isinstance(start_date, str):
                    # Try to parse with year, or use target_year/current year
                    try:
                        parsed_start = pd.to_datetime(start_date)
                    except:
                        # Add year context if missing
                        year = target_year or series.index[-1].year
                        parsed_start = pd.to_datetime(f"{start_date} {year}")
                else:
                    parsed_start = pd.to_datetime(start_date)
                
                if isinstance(end_date, str):
                    try:
                        parsed_end = pd.to_datetime(end_date)
                    except:
                        year = target_year or series.index[-1].year
                        parsed_end = pd.to_datetime(f"{end_date} {year}")
                else:
                    parsed_end = pd.to_datetime(end_date)
                
                # Make end_date inclusive for the whole day
                if parsed_end.hour == 0 and parsed_end.minute == 0:
                    parsed_end = parsed_end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                logger.info(f"Date range backtest: {parsed_start} to {parsed_end}")
                
                # Get context (all data before start_date) and target data (start to end)
                context_series = series[series.index < parsed_start]
                target_series = series[(series.index >= parsed_start) & (series.index <= parsed_end)]
                
                if len(context_series) < 3:
                    return {
                        "success": False,
                        "error": f"Not enough context data before {parsed_start.strftime('%Y-%m-%d')} (need at least 3 points, got {len(context_series)})"
                    }
                
                if len(target_series) == 0:
                    return {
                        "success": False,
                        "error": f"No data found between {parsed_start.strftime('%Y-%m-%d')} and {parsed_end.strftime('%Y-%m-%d')}"
                    }
                
                pred_horizon = len(target_series)
                window_desc = f"all data before {parsed_start.strftime('%Y-%m-%d')}"
                
                logger.info(f"Date-range backtest: Context: {len(context_series)} points, Target: {parsed_start.strftime('%Y-%m-%d')} to {parsed_end.strftime('%Y-%m-%d')} ({pred_horizon} points)")
                
                actual_series = target_series if compare_actual else None
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not parse date range: {e}"
                }
        
        # Handle month range (start_month to end_month)
        elif start_month is not None and end_month is not None:
            # Parse start month
            if isinstance(start_month, int):
                start_month_num = start_month
            else:
                start_month_num = parse_month_name(str(start_month))
                if start_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse start_month: '{start_month}'"
                    }
            
            # Parse end month
            if isinstance(end_month, int):
                end_month_num = end_month
            else:
                end_month_num = parse_month_name(str(end_month))
                if end_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse end_month: '{end_month}'"
                    }
            
            # Determine the year - find the most relevant year with this month range
            year = target_year
            if year is None:
                # Find all years that have the start month
                years_with_start = series.index[series.index.month == start_month_num].year.unique()
                if len(years_with_start) == 0:
                    return {
                        "success": False,
                        "error": f"No data found for {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)}"
                    }
                # Use the latest year that has both start and end months
                for yr in sorted(years_with_start, reverse=True):
                    if (series.index.month == end_month_num).any() and (series.index.year == yr).any():
                        year = int(yr)
                        break
                if year is None:
                    year = int(years_with_start[-1])
            
            # Get context (all data before start month) and target data (start to end month)
            context_series = get_context_before_month_range(series, start_month_num, year)
            target_series = filter_series_by_month_range(series, start_month_num, end_month_num, year)
            
            if len(context_series) < 3:
                return {
                    "success": False,
                    "error": f"Not enough context data before {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} {year} (need at least 3 points, got {len(context_series)})"
                }
            
            if len(target_series) == 0:
                return {
                    "success": False,
                    "error": f"No data found for {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} to {MONTH_NUMBER_TO_NAME.get(end_month_num, end_month_num)} {year}"
                }
            
            # Override context_end_idx and prediction horizon
            pred_horizon = len(target_series)
            window_desc = f"all data before {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} {year}"
            
            logger.info(f"Month-range backtest: Context: {len(context_series)} points, Target: {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} to {MONTH_NUMBER_TO_NAME.get(end_month_num, end_month_num)} {year} ({pred_horizon} points)")
            
            # Use the month-range context and target
            actual_series = target_series if compare_actual else None
            
        # Handle single month targeting
        elif target_month is not None:
            # Parse the month reference
            if isinstance(target_month, int):
                month_num = target_month
                month_spec = MONTH_NUMBER_TO_NAME.get(target_month, str(target_month))
                if target_year:
                    month_spec = f"{month_spec} {target_year}"
            else:
                month_spec = str(target_month)
                month_num = parse_month_name(month_spec)
            
            month_info = parse_month_reference(month_spec, series)
            
            if not month_info.get("success"):
                return {
                    "success": False,
                    "error": month_info.get("error", "Could not parse month"),
                    "available_months": month_info.get("available_months", [])
                }
            
            # Check for ambiguity
            if month_info.get("ambiguous") and target_year is None:
                return build_clarification_response(
                    month_info["matches"], 
                    month_info["month_name"]
                )
            
            # Use the first (or only) match
            match = month_info["matches"][0]
            month_num = month_info["month"]
            year = target_year or match["year"]
            
            # Get context (all data before target month) and target data
            context_series = get_context_before_month(series, month_num, year)
            target_series = filter_series_by_month(series, month_num, year)
            
            if len(context_series) < 3:
                return {
                    "success": False,
                    "error": f"Not enough context data before {MONTH_NUMBER_TO_NAME[month_num]} {year} (need at least 3 points, got {len(context_series)})"
                }
            
            if len(target_series) == 0:
                return {
                    "success": False,
                    "error": f"No data found for {MONTH_NUMBER_TO_NAME[month_num]} {year}"
                }
            
            # Override context_end_idx and prediction horizon
            pred_horizon = len(target_series)
            window_desc = f"all data before {MONTH_NUMBER_TO_NAME[month_num]} {year}"
            
            logger.info(f"Month-based backtest: Context: {len(context_series)} points, Target: {MONTH_NUMBER_TO_NAME[month_num]} {year} ({pred_horizon} points)")
            
            # Use the month-based context and target
            actual_series = target_series if compare_actual else None
            
        else:
            # Original logic: Parse context_end specification
            if context_end is None:
                # Default: use 80% as context
                context_end_idx = int(series_length * 0.8)
                window_desc = f"first {context_end_idx} points (80% of data)"
            else:
                context_end_idx, window_desc = parse_window_specification(context_end, series, series_length)
            
            # Ensure we have enough context
            if context_end_idx < 3:
                return {
                    "success": False,
                    "error": f"Context window too small (need at least 3 points, got {context_end_idx})"
                }
            
            # Determine prediction horizon
            remaining_points = series_length - context_end_idx
            if horizon is None:
                # Predict all remaining points
                pred_horizon = remaining_points
            else:
                pred_horizon = min(horizon, remaining_points) if compare_actual else horizon
            
            if pred_horizon < 1:
                return {
                    "success": False,
                    "error": "No points to predict. Try a smaller context window."
                }
            
            # Split the data
            context_series = series.iloc[:context_end_idx]
            actual_series = series.iloc[context_end_idx:context_end_idx + pred_horizon] if compare_actual else None
            
            logger.info(f"Context: {len(context_series)} points, Horizon: {pred_horizon}")
        
        # Use the forecasting tool's internal logic
        from ..forecasters import create_forecaster, get_available_forecasters
        
        # Run forecast
        if forecaster == "multi":
            # Multi-model ensemble
            available = get_available_forecasters()
            models_to_use = _select_forecast_models(context_series, available)
        else:
            models_to_use = [forecaster] if isinstance(forecaster, str) else forecaster
        
        # Models accepting model_size
        MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}
        
        all_forecasts = []
        models_succeeded = []
        forecast_timestamps = None
        
        for model_name in models_to_use:
            try:
                if model_name in MODELS_WITH_SIZE:
                    fc = create_forecaster(model_name, model_size=model_size)
                else:
                    fc = create_forecaster(model_name)
                
                if fc is None:
                    continue
                
                result = fc.forecast(
                    series=context_series, 
                    horizon=pred_horizon, 
                    context_length=len(context_series)
                )
                
                if result.success:
                    models_succeeded.append(model_name)
                    all_forecasts.append(result.forecast)
                    if forecast_timestamps is None:
                        forecast_timestamps = result.timestamps
                        
            except Exception as e:
                logger.warning(f"Forecaster {model_name} failed: {e}")
                continue
        
        if not all_forecasts:
            return {"success": False, "error": "All forecasters failed"}
        
        # Compute ensemble prediction
        forecast_array = np.array(all_forecasts)
        ensemble_mean = np.mean(forecast_array, axis=0).tolist()
        ensemble_std = np.std(forecast_array, axis=0).tolist()
        
        # Build response
        result = {
            "success": True,
            "context_data": {
                "timestamps": [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in context_series.index],
                "values": context_series.tolist(),
            },
            "predictions": {
                "timestamps": forecast_timestamps if forecast_timestamps else [],
                "values": ensemble_mean,
            },
            "horizon": pred_horizon,
            "models_used": models_succeeded,
            "window_description": f"Used {window_desc} to predict next {pred_horizon} points",
        }
        
        # Add actuals and metrics if comparing
        if compare_actual and actual_series is not None and len(actual_series) > 0:
            actual_values = actual_series.tolist()
            actual_timestamps = [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in actual_series.index]
            
            result["actuals"] = {
                "timestamps": actual_timestamps,
                "values": actual_values,
            }
            
            # Calculate metrics
            min_len = min(len(ensemble_mean), len(actual_values))
            if min_len > 0:
                metrics = calculate_all_metrics(
                    actual_values[:min_len],
                    ensemble_mean[:min_len]
                )
                result["metrics"] = metrics
                result["metrics_description"] = (
                    f"MAE: {metrics['mae']:.4f} | "
                    f"MAPE: {metrics['mape']:.2f}% | "
                    f"SMAPE: {metrics['smape']:.2f}% | "
                    f"RMSE: {metrics['rmse']:.4f}"
                )
        
        return _to_python_types(result)
        
    except Exception as e:
        import traceback
        logger.error(f"Backtest error: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _select_forecast_models(series: pd.Series, available: List[str]) -> List[str]:
    """Select appropriate forecast models based on data characteristics."""
    selected = []
    
    # Always include baseline
    if "zscore_baseline" in available:
        selected.append("zscore_baseline")
    elif "baseline" in available:
        selected.append("baseline")
    
    # Prefer foundation models
    priority_models = ["moirai", "chronos", "timesfm", "tirex", "lag-llama"]
    for model in priority_models:
        if model in available and len(selected) < 5:
            selected.append(model)
    
    # Ensure at least 2 models
    for model in available:
        if model not in selected and model != "llm" and len(selected) < 2:
            selected.append(model)
    
    return selected[:5]


# Tool registration info
TOOL_INFO = {
    "name": "backtest_forecast",
    "function": backtest_forecast,
    "description": "Test forecasts on historical data with custom context windows and compare with actual values. Supports specific date ranges (start_date/end_date), month ranges (start_month/end_month), or single month (target_month). Returns accuracy metrics (MAE, MAPE, SMAPE, RMSE).",
    "parameters": {
        "csv_path": "Path to CSV file with time series",
        "target_col": "Name of column to forecast",
        "context_end": "Where context ends: integer, 'first 5', 'first half', '80%', etc.",
        "horizon": "Steps to predict (default: predict remaining data)",
        "forecaster": "'multi' for ensemble (default), or specific forecaster name",
        "compare_actual": "Compare with actual values (default True)",
        "target_month": "Single target month: integer (1-12) or string ('September', 'Sep')",
        "target_year": "Target year if month/date is ambiguous",
        "start_month": "Start of month range for date ranges like 'Sept to Dec'. Integer (1-12) or string.",
        "end_month": "End of month range for date ranges like 'Sept to Dec'. Integer (1-12) or string.",
        "start_date": "Start date for specific date ranges like 'Jan 4 to Jan 11'. String (e.g., 'Jan 4', '2026-01-04').",
        "end_date": "End date for specific date ranges like 'Jan 4 to Jan 11'. String (e.g., 'Jan 11', '2026-01-11').",
    }
}
