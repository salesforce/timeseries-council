# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Download utilities for exporting time series analysis results.
Supports CSV and JSON formats for various data types.
"""

import csv
import json
import io
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime


def get_timestamp_str() -> str:
    """Get a timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_forecast_csv(
    data: Dict[str, Any],
    include_confidence: bool = True,
    historical_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format forecast data as CSV.
    
    Args:
        data: Forecast result data containing predictions, timestamps, bounds
        include_confidence: Whether to include confidence intervals
        historical_data: Optional historical data to prepend
        
    Returns:
        CSV string
    """
    output = io.StringIO()
    
    # Handle various forecast structures:
    # 1. Backtest: {"predictions": {"values": [...], "timestamps": [...], ...}}
    # 2. Regular forecast: {"predictions": [...], "timestamps": [...], ...}
    # 3. Ensemble: {"forecast": [...], "timestamps": [...], "uncertainty": [...], ...}
    
    predictions_obj = data.get("predictions", {})
    
    # Check if predictions is a dict (backtest) or list (regular forecast)
    if isinstance(predictions_obj, dict) and predictions_obj.get("values"):
        # Backtest format - nested structure
        predictions = predictions_obj.get("values", [])
        timestamps = predictions_obj.get("timestamps", [])
        lower_bounds = predictions_obj.get("lower_bound", [])
        upper_bounds = predictions_obj.get("upper_bound", [])
    elif isinstance(predictions_obj, list) and len(predictions_obj) > 0:
        # Regular forecast with predictions list
        predictions = predictions_obj
        timestamps = data.get("timestamps") or data.get("forecast_timestamps") or list(range(len(predictions)))
        lower_bounds = data.get("lower_bound", [])
        upper_bounds = data.get("upper_bound", [])
    else:
        # Ensemble format - use forecast field
        predictions = data.get("forecast", [])
        timestamps = data.get("timestamps") or data.get("forecast_timestamps") or list(range(len(predictions)))
        # For ensemble, compute bounds from uncertainty if no explicit bounds
        uncertainty = data.get("uncertainty", [])
        lower_bounds = data.get("lower_bound", [])
        upper_bounds = data.get("upper_bound", [])
        if not lower_bounds and uncertainty:
            lower_bounds = [p - u for p, u in zip(predictions, uncertainty)]
        if not upper_bounds and uncertainty:
            upper_bounds = [p + u for p, u in zip(predictions, uncertainty)]
    
    # Also check for context_data (backtest format)
    context_data = data.get("context_data", {})
    actuals_data = data.get("actuals", {})
    
    # Determine columns - simplified without bounds per user request
    has_actuals = bool(actuals_data.get("values"))
    
    fieldnames = ["timestamp", "value", "type"]
    if has_actuals:
        fieldnames.append("actual")
    if data.get("model"):
        fieldnames.append("model")
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    model_name = data.get("model", "")
    
    # Add context data if available (backtest)
    if context_data.get("values"):
        ctx_timestamps = context_data.get("timestamps", list(range(len(context_data["values"]))))
        for ts, val in zip(ctx_timestamps, context_data["values"]):
            row = {
                "timestamp": ts,
                "value": val,
                "type": "context"
            }
            if has_actuals:
                row["actual"] = ""
            if data.get("model"):
                row["model"] = ""
            writer.writerow(row)
    
    # Add historical data if provided (regular forecast)
    elif historical_data and historical_data.get("values"):
        hist_timestamps = historical_data.get("timestamps", list(range(len(historical_data["values"]))))
        for ts, val in zip(hist_timestamps, historical_data["values"]):
            row = {
                "timestamp": ts,
                "value": val,
                "type": "historical"
            }
            if has_actuals:
                row["actual"] = ""
            if data.get("model"):
                row["model"] = ""
            writer.writerow(row)
    
    # Add predictions
    actual_values = actuals_data.get("values", []) if has_actuals else []
    
    for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
        row = {
            "timestamp": ts,
            "value": pred,
            "type": "forecast"
        }
        if has_actuals:
            row["actual"] = actual_values[i] if i < len(actual_values) else ""
        if data.get("model"):
            row["model"] = model_name
        writer.writerow(row)
    
    return output.getvalue()


def format_forecast_json(
    data: Dict[str, Any],
    include_confidence: bool = True,
    historical_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format forecast data as JSON structure."""
    result = {
        "export_type": "forecast",
        "exported_at": datetime.now().isoformat(),
        "model": data.get("model", "unknown"),
    }
    
    # Handle various forecast structures (same logic as CSV)
    predictions_obj = data.get("predictions", {})
    
    if isinstance(predictions_obj, dict) and predictions_obj.get("values"):
        # Backtest format - nested structure
        predictions = predictions_obj.get("values", [])
        timestamps = predictions_obj.get("timestamps", [])
        lower_bounds = predictions_obj.get("lower_bound", [])
        upper_bounds = predictions_obj.get("upper_bound", [])
    elif isinstance(predictions_obj, list) and len(predictions_obj) > 0:
        # Regular forecast with predictions list
        predictions = predictions_obj
        timestamps = data.get("timestamps") or data.get("forecast_timestamps") or list(range(len(predictions)))
        lower_bounds = data.get("lower_bound", [])
        upper_bounds = data.get("upper_bound", [])
    else:
        # Ensemble format - use forecast field
        predictions = data.get("forecast", [])
        timestamps = data.get("timestamps") or data.get("forecast_timestamps") or list(range(len(predictions)))
        uncertainty = data.get("uncertainty", [])
        lower_bounds = data.get("lower_bound", [])
        upper_bounds = data.get("upper_bound", [])
        if not lower_bounds and uncertainty:
            lower_bounds = [p - u for p, u in zip(predictions, uncertainty)]
        if not upper_bounds and uncertainty:
            upper_bounds = [p + u for p, u in zip(predictions, uncertainty)]
    
    result["forecast"] = {
        "timestamps": timestamps,
        "values": predictions,
        "count": len(predictions)
    }
    
    if include_confidence and lower_bounds and upper_bounds:
        result["forecast"]["confidence_interval"] = {
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds
        }
    
    # Add context data if available (backtest)
    context_data = data.get("context_data", {})
    if context_data.get("values"):
        result["context"] = {
            "timestamps": context_data.get("timestamps", []),
            "values": context_data.get("values", []),
            "count": len(context_data.get("values", []))
        }
    
    # Add actuals if available (backtest comparison)
    actuals_data = data.get("actuals", {})
    if actuals_data.get("values"):
        result["actuals"] = {
            "timestamps": actuals_data.get("timestamps", []),
            "values": actuals_data.get("values", []),
            "count": len(actuals_data.get("values", []))
        }
    
    # Add metrics if available (backtest)
    if data.get("metrics"):
        result["metrics"] = data["metrics"]
    
    if historical_data:
        result["historical"] = {
            "timestamps": historical_data.get("timestamps", []),
            "values": historical_data.get("values", []),
            "count": len(historical_data.get("values", []))
        }
    
    return result


def format_anomalies_csv(
    data: Dict[str, Any],
    include_flags: bool = True,
    historical_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format anomaly detection results as CSV.
    
    Args:
        data: Anomaly result data
        include_flags: Whether to include confidence flags
        historical_data: Optional full time series for context
        
    Returns:
        CSV string
    """
    output = io.StringIO()
    
    fieldnames = ["timestamp", "value", "anomaly_score"]
    if include_flags:
        fieldnames.extend(["confidence", "is_anomaly", "detected_by_models"])
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    # Combine all anomaly sources
    all_anomalies = []
    
    # High confidence anomalies
    for anomaly in data.get("high_confidence_anomalies", []):
        all_anomalies.append({
            "timestamp": anomaly.get("timestamp", ""),
            "value": anomaly.get("value", ""),
            "anomaly_score": anomaly.get("score", ""),
            "confidence": "high",
            "is_anomaly": True,
            "detected_by_models": anomaly.get("detected_by", "")
        })
    
    # Medium confidence anomalies
    for anomaly in data.get("medium_confidence_anomalies", []):
        all_anomalies.append({
            "timestamp": anomaly.get("timestamp", ""),
            "value": anomaly.get("value", ""),
            "anomaly_score": anomaly.get("score", ""),
            "confidence": "medium",
            "is_anomaly": True,
            "detected_by_models": anomaly.get("detected_by", "")
        })
    
    # Generic anomalies list
    for anomaly in data.get("anomalies", []):
        if not any(a["timestamp"] == anomaly.get("timestamp") for a in all_anomalies):
            all_anomalies.append({
                "timestamp": anomaly.get("timestamp", ""),
                "value": anomaly.get("value", ""),
                "anomaly_score": anomaly.get("score", ""),
                "confidence": anomaly.get("confidence", ""),
                "is_anomaly": True,
                "detected_by_models": anomaly.get("detected_by", "")
            })
    
    # Sort by timestamp if possible
    try:
        all_anomalies.sort(key=lambda x: x["timestamp"])
    except:
        pass
    
    for anomaly in all_anomalies:
        row = {
            "timestamp": anomaly["timestamp"],
            "value": anomaly["value"],
            "anomaly_score": anomaly["anomaly_score"]
        }
        if include_flags:
            row["confidence"] = anomaly["confidence"]
            row["is_anomaly"] = anomaly["is_anomaly"]
            row["detected_by_models"] = anomaly["detected_by_models"]
        writer.writerow(row)
    
    return output.getvalue()


def format_anomalies_json(
    data: Dict[str, Any],
    include_flags: bool = True,
    historical_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format anomaly detection results as JSON."""
    result = {
        "export_type": "anomaly_detection",
        "exported_at": datetime.now().isoformat(),
        "summary": {
            "total_anomalies": data.get("anomaly_count", 0),
            "high_confidence_count": len(data.get("high_confidence_anomalies", [])),
            "medium_confidence_count": len(data.get("medium_confidence_anomalies", [])),
            "models_used": data.get("models_used", [])
        }
    }
    
    if include_flags:
        result["high_confidence_anomalies"] = data.get("high_confidence_anomalies", [])
        result["medium_confidence_anomalies"] = data.get("medium_confidence_anomalies", [])
    
    result["all_anomalies"] = data.get("anomalies", [])
    
    if historical_data:
        result["time_series_context"] = {
            "total_points": len(historical_data.get("values", [])),
            "time_range": {
                "start": historical_data.get("timestamps", [""])[0] if historical_data.get("timestamps") else "",
                "end": historical_data.get("timestamps", [""])[-1] if historical_data.get("timestamps") else ""
            }
        }
    
    return result


def format_multimodel_csv(data: Dict[str, Any]) -> str:
    """
    Format multi-model comparison results as flat CSV.
    Each row has timestamp and predictions from each model.
    """
    output = io.StringIO()
    
    model_results = data.get("model_results", {})
    if not model_results:
        return ""
    
    # Get all model names
    model_names = list(model_results.keys())
    fieldnames = ["timestamp"] + model_names
    
    # Add ensemble columns if aggregated data exists
    aggregated = data.get("aggregated", {})
    if aggregated.get("mean_prediction"):
        fieldnames.append("ensemble_mean")
    if aggregated.get("median_prediction"):
        fieldnames.append("ensemble_median")
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    # Find the maximum length of predictions
    max_len = 0
    model_predictions = {}
    model_timestamps = {}
    
    for model_name, result in model_results.items():
        if result.get("success") and result.get("data"):
            preds = result["data"].get("predictions", [])
            ts = result["data"].get("timestamps", list(range(len(preds))))
            model_predictions[model_name] = preds
            model_timestamps[model_name] = ts
            max_len = max(max_len, len(preds))
    
    # Use first model's timestamps as reference
    ref_timestamps = list(model_timestamps.values())[0] if model_timestamps else list(range(max_len))
    
    for i in range(max_len):
        row = {"timestamp": ref_timestamps[i] if i < len(ref_timestamps) else i}
        
        for model_name in model_names:
            preds = model_predictions.get(model_name, [])
            row[model_name] = preds[i] if i < len(preds) else ""
        
        if aggregated.get("mean_prediction"):
            mean_preds = aggregated["mean_prediction"]
            row["ensemble_mean"] = mean_preds[i] if i < len(mean_preds) else ""
        
        if aggregated.get("median_prediction"):
            median_preds = aggregated["median_prediction"]
            row["ensemble_median"] = median_preds[i] if i < len(median_preds) else ""
        
        writer.writerow(row)
    
    return output.getvalue()


def format_multimodel_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format multi-model comparison results as JSON."""
    result = {
        "export_type": "multi_model_comparison",
        "exported_at": datetime.now().isoformat(),
        "models": {}
    }
    
    model_results = data.get("model_results", {})
    
    for model_name, model_result in model_results.items():
        if model_result.get("success") and model_result.get("data"):
            result["models"][model_name] = {
                "predictions": model_result["data"].get("predictions", []),
                "timestamps": model_result["data"].get("timestamps", []),
                "success": True
            }
            if model_result["data"].get("lower_bound"):
                result["models"][model_name]["confidence_interval"] = {
                    "lower_bound": model_result["data"]["lower_bound"],
                    "upper_bound": model_result["data"].get("upper_bound", [])
                }
        else:
            result["models"][model_name] = {
                "success": False,
                "error": model_result.get("error", "Unknown error")
            }
    
    if data.get("aggregated"):
        result["ensemble"] = {
            "mean_prediction": data["aggregated"].get("mean_prediction", []),
            "median_prediction": data["aggregated"].get("median_prediction", []),
            "min_prediction": data["aggregated"].get("min_prediction", []),
            "max_prediction": data["aggregated"].get("max_prediction", [])
        }
    
    return result


def format_decomposition_csv(data: Dict[str, Any]) -> str:
    """Format time series decomposition results as CSV."""
    output = io.StringIO()
    
    # Check for chart_series format (new) or individual components
    chart_series = data.get("chart_series", {})
    
    if chart_series:
        # New format with chart_series
        fieldnames = ["timestamp"]
        components = {}
        ref_timestamps = None
        
        for key, series in chart_series.items():
            if series.get("values"):
                fieldnames.append(key)
                components[key] = series["values"]
                if ref_timestamps is None and series.get("timestamps"):
                    ref_timestamps = series["timestamps"]
        
        if not ref_timestamps and components:
            ref_timestamps = list(range(len(list(components.values())[0])))
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(ref_timestamps) if ref_timestamps else 0):
            row = {"timestamp": ref_timestamps[i]}
            for key in components:
                vals = components[key]
                row[key] = vals[i] if i < len(vals) else ""
            writer.writerow(row)
    else:
        # Legacy format with trend_data
        trend_data = data.get("trend_data", {})
        if trend_data.get("values"):
            fieldnames = ["timestamp", "trend"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            timestamps = trend_data.get("timestamps", list(range(len(trend_data["values"]))))
            for ts, val in zip(timestamps, trend_data["values"]):
                writer.writerow({"timestamp": ts, "trend": val})
    
    return output.getvalue()


def format_decomposition_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format time series decomposition results as JSON."""
    result = {
        "export_type": "decomposition",
        "exported_at": datetime.now().isoformat(),
        "components": {}
    }
    
    chart_series = data.get("chart_series", {})
    
    if chart_series:
        for key, series in chart_series.items():
            result["components"][key] = {
                "name": series.get("name", key),
                "timestamps": series.get("timestamps", []),
                "values": series.get("values", []),
                "description": series.get("description", "")
            }
    else:
        trend_data = data.get("trend_data", {})
        if trend_data:
            result["components"]["trend"] = {
                "name": trend_data.get("name", "Trend"),
                "timestamps": trend_data.get("timestamps", []),
                "values": trend_data.get("values", [])
            }
    
    return result


def format_historical_csv(data: Dict[str, Any]) -> str:
    """Format historical time series data as CSV."""
    output = io.StringIO()
    
    timestamps = data.get("timestamps", [])
    values = data.get("values", [])
    target_col = data.get("target_col", "value")
    date_col = data.get("date_col", "timestamp")
    
    fieldnames = [date_col, target_col]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for ts, val in zip(timestamps, values):
        writer.writerow({date_col: ts, target_col: val})
    
    return output.getvalue()


def format_historical_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format historical time series data as JSON."""
    return {
        "export_type": "historical_data",
        "exported_at": datetime.now().isoformat(),
        "target_column": data.get("target_col", "value"),
        "date_column": data.get("date_col", "timestamp"),
        "data": {
            "timestamps": data.get("timestamps", []),
            "values": data.get("values", [])
        },
        "summary": {
            "total_points": len(data.get("values", [])),
            "time_range": {
                "start": data.get("timestamps", [""])[0] if data.get("timestamps") else "",
                "end": data.get("timestamps", [""])[-1] if data.get("timestamps") else ""
            }
        }
    }


def format_full_report_json(
    skill_result: Dict[str, Any],
    historical_data: Optional[Dict[str, Any]] = None,
    session_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format a full analysis report as JSON."""
    result = {
        "export_type": "full_report",
        "exported_at": datetime.now().isoformat(),
        "session_info": session_info or {},
        "skill_executed": skill_result.get("skill_name", "unknown"),
        "execution_time": skill_result.get("execution_time"),
        "models_used": skill_result.get("models_used", []),
        "success": skill_result.get("success", False)
    }
    
    # Include the raw skill result data
    if skill_result.get("data"):
        result["analysis_results"] = skill_result["data"]
    
    # Include historical context
    if historical_data:
        result["historical_data"] = {
            "timestamps": historical_data.get("timestamps", []),
            "values": historical_data.get("values", []),
            "target_column": historical_data.get("target_col", "value")
        }
    
    return result
