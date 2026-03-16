# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Anomaly detection tool for the orchestrator.
Multi-model approach using 3-5 detectors for robust results.
"""

from typing import Dict, Any, Optional, List
from collections import Counter
import pandas as pd
import numpy as np

from ..logging import get_logger
from ..detectors import create_detector, get_available_detectors

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
    return obj

# Model selection rationale
MODEL_RATIONALE = {
    "zscore": "Z-Score: Fast statistical method, good baseline for normally distributed data",
    "mad": "MAD: Robust to existing outliers, works well with non-normal distributions",
    "isolation-forest": "Isolation Forest: ML-based, detects contextual anomalies via isolation",
    "lof": "LOF: Density-based detection, finds local outliers in varying density data",
    "windstats": "WindStats (Merlion): Window-based statistical detection for regular patterns",
    "spectral": "Spectral Residual (Merlion): Frequency-domain detection for periodic anomalies",
    "prophet": "Prophet (Merlion): Forecast-based detection for seasonal & trend changes",
    "lstm-vae": "LSTM-VAE: Deep learning detector for complex temporal patterns",
    "moirai": "Moirai: Foundation model back-prediction with probabilistic confidence intervals",
    "ecod": "ECOD: Parameter-free empirical CDF-based detection (PyOD, ADBench top performer)",
    "copod": "COPOD: Parameter-free copula-based detection for multivariate anomalies (PyOD)",
    "hbos": "HBOS: Very fast histogram-based outlier scoring (PyOD)",
    "knn": "KNN: Distance-based k-nearest neighbors anomaly detection (PyOD)",
    "ocsvm": "OCSVM: One-Class SVM, kernel-based boundary learning (PyOD)",
    "loda": "LODA: Lightweight ensemble of random projection histograms (PyOD)",
}

# Model priority for selection (statistical → ML → PyOD → advanced)
DETECTOR_PRIORITY = [
    # Statistical baselines (always try)
    "zscore", "mad",
    # ML-based (sklearn)
    "isolation-forest", "lof",
    # PyOD detectors (top ADBench performers)
    "ecod", "copod", "hbos", "knn", "loda",
    # Merlion-based (advanced)
    "windstats", "spectral", "prophet",
    # Deep learning
    "lstm-vae",
    # Foundation model (powerful but slower)
    "moirai",
    # PyOD (slower)
    "ocsvm",
]


def detect_anomalies(
    csv_path: str = None,
    target_col: str = None,
    detector: str = "multi",
    sensitivity: float = 2.0,
    target_month: any = None,
    target_year: int = None,
    start_date: str = None,
    end_date: str = None,
    start_month: any = None,
    end_month: any = None,
    series: pd.Series = None,
    memory: "DetectionMemory" = None,
    custom_prompt: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect anomalies in a time series using multiple models.

    Args:
        csv_path: Path to CSV file
        target_col: Column to analyze
        detector: 'multi' for multi-model (default), or specific detector name
        sensitivity: Detection sensitivity (higher = fewer anomalies)
        target_month: Filter to specific month (1-12 or "September", "Sep", etc.)
        target_year: Filter to specific year (required if month is ambiguous)
        start_date: Start date for date range (e.g., "15th Aug", "Aug 15", "2024-08-15")
        end_date: End date for date range (e.g., "3rd Sept", "Sept 3", "2024-09-03")
        start_month: Start month for month range (e.g., "April" or 4 for Q2)
        end_month: End month for month range (e.g., "June" or 6 for Q2)
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)
        memory: Optional DetectionMemory with previous anomalies, baseline stats,
                and domain context for informed detection
        **kwargs: Additional detector-specific arguments

    Returns:
        Dict with anomaly information from multiple models
    """
    # Handle month range filtering (e.g., Q2 = April to June)
    if start_month is not None and end_month is not None:
        kwargs['start_month'] = start_month
        kwargs['end_month'] = end_month
        kwargs['target_year'] = target_year
    # Handle date range filtering
    elif start_date is not None and end_date is not None:
        kwargs['start_date'] = start_date
        kwargs['end_date'] = end_date
        kwargs['target_year'] = target_year
    # Handle single month filtering
    elif target_month is not None:
        kwargs['target_month'] = target_month
        kwargs['target_year'] = target_year
    
    # Pass memory through kwargs for sub-functions
    if memory is not None:
        kwargs['memory'] = memory

    # Pass custom_prompt to single detector if provided
    if custom_prompt is not None:
        kwargs['custom_prompt'] = custom_prompt

    # If specific detector requested (not multi), use single model
    if detector != "multi" and detector.lower() != "ensemble":
        return _detect_single(csv_path, target_col, detector, sensitivity, series=series, **kwargs)

    # Multi-model detection
    return _detect_multi_model(csv_path, target_col, sensitivity, series=series, **kwargs)


def _detect_single(
    csv_path: str,
    target_col: str,
    detector: str,
    sensitivity: float,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """Run detection with a single model."""
    logger.info(f"Detecting anomalies: {detector} on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series
        from ..utils.date_parsing import (
            parse_month_reference,
            filter_series_by_month,
            build_clarification_response,
            MONTH_NUMBER_TO_NAME,
            parse_month_name,
            parse_date_range,
            get_context_before_date,
            filter_series_by_date_range,
            get_context_before_month
        )

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 3:
            return {
                "success": False,
                "error": "Need at least 3 numeric data points for detection"
            }

        # Extract memory before passing kwargs to detector constructor
        memory = kwargs.pop('memory', None)

        # Handle date range filtering - most specific option
        start_month = kwargs.pop('start_month', None)
        end_month = kwargs.pop('end_month', None)
        start_date = kwargs.pop('start_date', None)
        end_date = kwargs.pop('end_date', None)
        target_month = kwargs.pop('target_month', None)
        target_year = kwargs.pop('target_year', None)

        range_filter_desc = None
        target_start_idx = None
        target_end_idx = None
        full_series = series  # Keep full series for context-aware detectors
        
        # Determine if this is a context-aware detector (needs full series + target indices)
        context_aware_detectors = {'moirai'}  # Detectors that use prediction-based anomaly detection
        is_context_aware = detector.lower() in context_aware_detectors
        
        # Month range filtering (e.g., Q2 = April to June)
        if start_month is not None and end_month is not None:
            from ..utils.date_parsing import (
                parse_month_name,
                filter_series_by_month_range,
                MONTH_NUMBER_TO_NAME
            )
            
            # Parse month names to numbers
            if isinstance(start_month, int):
                start_month_num = start_month
            else:
                start_month_num = parse_month_name(str(start_month))
                if start_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse start_month: '{start_month}'"
                    }
            
            if isinstance(end_month, int):
                end_month_num = end_month
            else:
                end_month_num = parse_month_name(str(end_month))
                if end_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse end_month: '{end_month}'"
                    }
            
            # Determine year - use most recent occurrence if not specified
            if target_year is None:
                # Find years that have data for the start month
                years_with_start = series.index[series.index.month == start_month_num].year.unique()
                if len(years_with_start) == 0:
                    return {
                        "success": False,
                        "error": f"No data found for {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)}"
                    }
                target_year = int(years_with_start.max())
            
            range_filter_desc = f"{MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} to {MONTH_NUMBER_TO_NAME.get(end_month_num, end_month_num)} {target_year}"
            
            if is_context_aware:
                # For context-aware detectors, keep full series and compute target indices
                # Create mask for the month range
                if start_month_num <= end_month_num:
                    # Normal range (e.g., April to June)
                    target_mask = (
                        (series.index.month >= start_month_num) & 
                        (series.index.month <= end_month_num) & 
                        (series.index.year == target_year)
                    )
                else:
                    # Wrap-around range (e.g., November to February)
                    target_mask = (
                        ((series.index.month >= start_month_num) | (series.index.month <= end_month_num)) & 
                        (series.index.year == target_year)
                    )
                
                if not target_mask.any():
                    return {
                        "success": False,
                        "error": f"No data found for {range_filter_desc}"
                    }
                
                target_indices = np.where(target_mask)[0]
                target_start_idx = int(target_indices[0])
                target_end_idx = int(target_indices[-1]) + 1  # Exclusive end
                
                logger.info(f"Context-aware detection: target indices {target_start_idx} to {target_end_idx} ({range_filter_desc})")
            else:
                # For statistical detectors, just filter to target range
                series = filter_series_by_month_range(series, start_month_num, end_month_num, target_year)
                
                if len(series) < 3:
                    return {
                        "success": False,
                        "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                    }
                
                logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")
        
        elif start_date is not None and end_date is not None:
            # Parse the date range
            date_range = parse_date_range(start_date, end_date, series, target_year)
            
            if not date_range.get("success"):
                return {
                    "success": False,
                    "error": date_range.get("error", "Could not parse date range")
                }
            
            parsed_start = date_range["start_date"]
            parsed_end = date_range["end_date"]
            range_filter_desc = date_range["description"]
            
            if is_context_aware:
                # For context-aware detectors, keep full series and compute target indices
                # Find indices for the target range
                target_mask = (series.index >= parsed_start) & (series.index <= parsed_end)
                if not target_mask.any():
                    return {
                        "success": False,
                        "error": f"No data found in range {range_filter_desc}"
                    }
                
                target_indices = np.where(target_mask)[0]
                target_start_idx = int(target_indices[0])
                target_end_idx = int(target_indices[-1]) + 1  # Exclusive end
                
                logger.info(f"Context-aware detection: target indices {target_start_idx} to {target_end_idx} ({range_filter_desc})")
            else:
                # For statistical detectors, just filter to target range
                series = filter_series_by_date_range(series, parsed_start, parsed_end)
                
                if len(series) < 3:
                    return {
                        "success": False,
                        "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                    }
                
                logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")
        
        elif target_month is not None:
            # Parse month reference
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
            
            month_num = month_info["month"]
            year = target_year or month_info["matches"][0]["year"]
            range_filter_desc = f"{MONTH_NUMBER_TO_NAME[month_num]} {year}"
            
            if is_context_aware:
                # For context-aware detectors, keep full series and compute target indices
                target_mask = (series.index.month == month_num) & (series.index.year == year)
                if not target_mask.any():
                    return {
                        "success": False,
                        "error": f"No data found for {range_filter_desc}"
                    }
                
                target_indices = np.where(target_mask)[0]
                target_start_idx = int(target_indices[0])
                target_end_idx = int(target_indices[-1]) + 1  # Exclusive end
                
                logger.info(f"Context-aware detection: target indices {target_start_idx} to {target_end_idx} ({range_filter_desc})")
            else:
                # For statistical detectors, just filter to target month
                series = filter_series_by_month(series, month_num, year)
                
                if len(series) < 3:
                    return {
                        "success": False, 
                        "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                    }
                
                logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")

        # Remove provider from kwargs - it's only used for LLM detector
        provider = kwargs.pop("provider", None)
        
        try:
            if detector.lower() == "llm" or detector.lower() == "rule":
                if not provider:
                    return {"success": False, "error": f"{detector} detector requires 'provider'"}
                det = create_detector(detector, provider=provider, **kwargs)
            else:
                det = create_detector(detector, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            return {"success": False, "error": f"Failed to create detector: {e}"}

        if det is None:
            return {"success": False, "error": f"Detector '{detector}' not available"}

        # Call detector with appropriate parameters
        if is_context_aware and target_start_idx is not None:
            # Pass full series with target indices for context-aware detectors
            result = det.detect(
                series=full_series,
                sensitivity=sensitivity,
                memory=memory,
                target_start_idx=target_start_idx,
                target_end_idx=target_end_idx
            )
        else:
            result = det.detect(series=series, sensitivity=sensitivity, memory=memory)

        if result.success:
            anomaly_list = []
            for a in result.anomalies:
                anomaly_info = {
                    "timestamp": a.timestamp,
                    "value": a.value,
                    "score": round(a.score, 2) if a.score else None,
                    "type": a.anomaly_type.value if a.anomaly_type else None,
                }
                if a.explanation:
                    anomaly_info["explanation"] = a.explanation
                anomaly_list.append(anomaly_info)

            mean_val = result.metadata.get("mean") if result.metadata else None
            std_val = result.metadata.get("std") if result.metadata else None

            return _to_python_types({
                "success": True,
                "values": series.tolist(),
                "timestamps": [str(idx) for idx in series.index],
                "anomaly_count": result.anomaly_count,
                "anomalies": anomaly_list,
                "sensitivity": result.threshold,
                "mean": round(float(mean_val), 2) if mean_val is not None else None,
                "std": round(float(std_val), 2) if std_val is not None else None,
                "model": result.detector_name,
                "models_used": [result.detector_name],
                "date_range_filter": range_filter_desc,
                "metadata": result.metadata
            })
        else:
            return {"success": False, "error": result.error}

    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _detect_multi_model(
    csv_path: str,
    target_col: str,
    sensitivity: float,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """Run detection with multiple models and aggregate results."""
    logger.info(f"Multi-model anomaly detection on {csv_path or 'series'}:{target_col or 'provided'}")

    # Extract LLM provider for intelligent model selection (injected by orchestrator)
    provider = kwargs.pop("provider", None)
    # Extract detection memory
    memory = kwargs.pop("memory", None)

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 3:
            return {"success": False, "error": "Need at least 3 numeric data points"}

        # Handle date range and month-based filtering
        start_month = kwargs.pop('start_month', None)
        end_month = kwargs.pop('end_month', None)
        start_date = kwargs.pop('start_date', None)
        end_date = kwargs.pop('end_date', None)
        target_month = kwargs.pop('target_month', None)
        target_year = kwargs.pop('target_year', None)
        range_filter_desc = None
        
        # Month range filtering takes priority (e.g., Q2 = April to June)
        if start_month is not None and end_month is not None:
            from ..utils.date_parsing import (
                parse_month_name,
                filter_series_by_month_range,
                MONTH_NUMBER_TO_NAME
            )
            
            # Parse month names to numbers
            if isinstance(start_month, int):
                start_month_num = start_month
            else:
                start_month_num = parse_month_name(str(start_month))
                if start_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse start_month: '{start_month}'"
                    }
            
            if isinstance(end_month, int):
                end_month_num = end_month
            else:
                end_month_num = parse_month_name(str(end_month))
                if end_month_num is None:
                    return {
                        "success": False,
                        "error": f"Could not parse end_month: '{end_month}'"
                    }
            
            # Determine year - use most recent occurrence if not specified
            if target_year is None:
                # Find years that have data for the start month
                years_with_start = series.index[series.index.month == start_month_num].year.unique()
                if len(years_with_start) == 0:
                    return {
                        "success": False,
                        "error": f"No data found for {MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)}"
                    }
                target_year = int(years_with_start.max())
            
            range_filter_desc = f"{MONTH_NUMBER_TO_NAME.get(start_month_num, start_month_num)} to {MONTH_NUMBER_TO_NAME.get(end_month_num, end_month_num)} {target_year}"
            
            # Filter to the target range
            series = filter_series_by_month_range(series, start_month_num, end_month_num, target_year)
            
            if len(series) < 3:
                return {
                    "success": False,
                    "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                }
            
            logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")
        
        # Date range filtering takes priority over month filtering
        elif start_date is not None and end_date is not None:
            from ..utils.date_parsing import (
                parse_date_range,
                filter_series_by_date_range
            )
            
            # Parse the date range
            date_range = parse_date_range(start_date, end_date, series, target_year)
            
            if not date_range.get("success"):
                return {
                    "success": False,
                    "error": date_range.get("error", "Could not parse date range")
                }
            
            parsed_start = date_range["start_date"]
            parsed_end = date_range["end_date"]
            range_filter_desc = date_range["description"]
            
            # Filter to the target range
            series = filter_series_by_date_range(series, parsed_start, parsed_end)
            
            if len(series) < 3:
                return {
                    "success": False,
                    "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                }
            
            logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")
        
        elif target_month is not None:
            from ..utils.date_parsing import (
                parse_month_reference,
                filter_series_by_month,
                build_clarification_response,
                MONTH_NUMBER_TO_NAME,
                parse_month_name
            )
            
            # Parse month reference
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
            
            # Filter to the target month
            month_num = month_info["month"]
            year = target_year or month_info["matches"][0]["year"]
            range_filter_desc = f"{MONTH_NUMBER_TO_NAME[month_num]} {year}"
            series = filter_series_by_month(series, month_num, year)
            
            if len(series) < 3:
                return {
                    "success": False, 
                    "error": f"Not enough data points in {range_filter_desc} (need at least 3, got {len(series)})"
                }
            
            logger.info(f"Filtered to {range_filter_desc}: {len(series)} points")

        # Select 3-5 models based on available detectors
        available = get_available_detectors()
        models_to_use = _select_detection_models(series, available, provider=provider)

        logger.info(f"Using {len(models_to_use)} detectors: {models_to_use}")

        # Run each detector
        all_results = []
        all_anomaly_timestamps = []
        model_details = []
        models_succeeded = []

        for model_name in models_to_use:
            try:
                det = create_detector(model_name, auto_setup=False)
                if det is None:
                    continue

                result = det.detect(series=series, sensitivity=sensitivity, memory=memory)

                if result.success:
                    models_succeeded.append(model_name)
                    anomalies_found = []
                    for a in result.anomalies:
                        anomalies_found.append(a.timestamp)
                        all_anomaly_timestamps.append(a.timestamp)

                    model_details.append({
                        "model": model_name,
                        "anomaly_count": result.anomaly_count,
                        "anomalies": anomalies_found,
                        "rationale": MODEL_RATIONALE.get(model_name, f"{model_name}: Specialized detector")
                    })

                    all_results.append({
                        "model": model_name,
                        "result": result,
                        "anomalies": result.anomalies
                    })
            except Exception as e:
                logger.warning(f"Detector {model_name} failed: {e}")
                continue

        if not all_results:
            return {"success": False, "error": "All detectors failed"}

        # Aggregate results - find consensus anomalies
        timestamp_counts = Counter(all_anomaly_timestamps)
        num_models = len(models_succeeded)

        # Anomalies detected by majority (>50%) of models
        consensus_threshold = max(2, num_models // 2 + 1)
        consensus_anomalies = []
        high_confidence = []
        medium_confidence = []

        for timestamp, count in timestamp_counts.items():
            confidence = count / num_models
            if count >= consensus_threshold:
                # Find the anomaly details from one of the results
                for r in all_results:
                    for a in r["anomalies"]:
                        if a.timestamp == timestamp:
                            anomaly_info = {
                                "timestamp": timestamp,
                                "value": a.value,
                                "score": round(a.score, 2) if a.score else None,
                                "type": a.anomaly_type.value if a.anomaly_type else None,
                                "detected_by": count,
                                "confidence": round(confidence, 2)
                            }
                            consensus_anomalies.append(anomaly_info)
                            if confidence >= 0.75:
                                high_confidence.append(anomaly_info)
                            else:
                                medium_confidence.append(anomaly_info)
                            break
                    else:
                        continue
                    break

        # Build model selection explanation
        model_selection_rationale = _build_model_rationale(series, models_succeeded)

        # Calculate agreement ratio
        if all_anomaly_timestamps:
            unique_anomalies = len(set(all_anomaly_timestamps))
            agreement_ratio = len(consensus_anomalies) / unique_anomalies if unique_anomalies > 0 else 0
        else:
            agreement_ratio = 1.0  # No anomalies found by any model

        return _to_python_types({
            "success": True,
            "values": series.tolist(),
            "timestamps": [str(idx) for idx in series.index],
            "anomaly_count": len(consensus_anomalies),
            "anomalies": consensus_anomalies,
            "high_confidence_anomalies": high_confidence,
            "medium_confidence_anomalies": medium_confidence,
            "sensitivity": sensitivity,
            "model": "Multi-Model Ensemble",
            "models_used": models_succeeded,
            "model_details": model_details,
            "model_selection_rationale": model_selection_rationale,
            "date_range_filter": range_filter_desc,
            "comparison": {
                "total_detections": len(all_anomaly_timestamps),
                "unique_anomalies": len(set(all_anomaly_timestamps)),
                "consensus_anomalies": len(consensus_anomalies),
                "agreement_ratio": round(float(agreement_ratio), 2),
                "consensus_threshold": f"{consensus_threshold}/{num_models} models"
            }
        })

    except Exception as e:
        logger.error(f"Multi-model detection error: {e}")
        return {"success": False, "error": str(e)}


def _select_detection_models(series: pd.Series, available: List[str], provider=None) -> List[str]:
    """Select 3-5 appropriate detection models based on data characteristics and availability.

    If an LLM provider is given, asks the LLM to choose models based on the
    series statistical profile. Falls back to the static DETECTOR_PRIORITY
    list when no provider is available or the LLM call fails.
    """
    # Try LLM-driven selection first
    if provider is not None:
        from .model_selector import llm_select_models
        llm_selected = llm_select_models(
            series=series,
            available=[d for d in available if d != "llm"],
            model_descriptions=MODEL_RATIONALE,
            provider=provider,
            task_type="anomaly detection",
        )
        if llm_selected:
            return llm_selected

    # Static fallback: priority order
    selected = []
    for detector in DETECTOR_PRIORITY:
        if detector in available and len(selected) < 5:
            selected.append(detector)

    # If we don't have enough, add any remaining available detectors
    for det in available:
        if det not in selected and det != "llm" and len(selected) < 5:
            selected.append(det)

    # Ensure we have at least 3 models (use what's available)
    if len(selected) < 3:
        logger.warning(f"Only {len(selected)} detectors available, need at least 3 for robust detection")

    return selected[:5]  # Cap at 5


def _build_model_rationale(series: pd.Series, models_used: List[str]) -> str:
    """Build explanation for why these models were selected."""
    rationale_parts = [
        f"Selected {len(models_used)} complementary detection methods for robust anomaly identification:\n"
    ]

    for model in models_used:
        if model in MODEL_RATIONALE:
            rationale_parts.append(f"• {MODEL_RATIONALE[model]}")
        else:
            rationale_parts.append(f"• {model}: Specialized anomaly detector")

    rationale_parts.append(
        f"\nConsensus approach: Anomalies flagged by majority of models are reported with confidence scores."
    )

    return "\n".join(rationale_parts)


# Tool registration info
TOOL_INFO = {
    "name": "detect_anomalies",
    "function": detect_anomalies,
    "description": "Find unusual spikes or drops using multiple ML models (3-5 detectors) for robust detection. Supports month range, date range, and single month filtering.",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to analyze",
        "detector": f"'multi' for ensemble (default), or specific: {', '.join(get_available_detectors())}",
        "sensitivity": "Threshold (default 2.0, lower = more sensitive)",
        "start_month": "Start month for month range (e.g., 'April' or 4 for Q2)",
        "end_month": "End month for month range (e.g., 'June' or 6 for Q2)",
        "target_month": "Single target month: integer (1-12) or string ('September', 'Sep')",
        "target_year": "Target year if month/date is ambiguous",
        "start_date": "Start date for date range (e.g., '15th Aug', 'Aug 15', '2024-08-15')",
        "end_date": "End date for date range (e.g., '3rd Sept', 'Sept 3', '2024-09-03')",
        "custom_prompt": "Custom business rules in natural language (REQUIRED if detector is 'rule')"
    }
}
