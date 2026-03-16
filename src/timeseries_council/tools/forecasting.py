# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Forecasting tool for the orchestrator.
Multi-model approach using 3-5 forecasters for robust predictions.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..logging import get_logger
from ..exceptions import ToolError
from ..forecasters import create_forecaster, get_available_forecasters

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
    "baseline": "Baseline: Fast statistical method using rolling statistics and trend extrapolation",
    "zscore_baseline": "Baseline: Statistical forecaster with trend detection, good for stable patterns",
    "moirai": "Moirai: Salesforce foundation model, strong on diverse time series patterns",
    "chronos": "Chronos: Amazon foundation model, excels at capturing complex seasonality with multivariate support",
    "timesfm": "TimesFM: Google foundation model, optimized for long-horizon forecasting",
    "lag-llama": "Lag-Llama: Probabilistic forecaster, provides uncertainty quantification",
    "tirex": "TiRex: NX-AI foundation model, strong quantile forecasting with context-based prediction",
    "llm": "LLM: Zero-shot forecasting using language models for contextual understanding",
}


def run_forecast(
    csv_path: str = None,
    target_col: str = None,
    horizon: int = 7,
    forecaster: str = "multi",
    model_size: str = "small",
    context_length: int = 168,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run forecast on a time series using multiple models.

    Args:
        csv_path: Path to CSV file with time series data
        target_col: Name of the column to forecast
        horizon: Number of steps to forecast
        forecaster: 'multi' for multi-model (default), specific forecaster name, or list of names
        model_size: Model size for applicable forecasters
        context_length: Number of historical points to use as context
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)
        **kwargs: Additional forecaster-specific arguments

    Returns:
        Dict with forecast results from multiple models
    """
    # Handle list of forecasters (e.g., ["moirai", "chronos"])
    if isinstance(forecaster, list):
        if len(forecaster) == 1:
            # Single model in list
            return _forecast_single(csv_path, target_col, horizon, forecaster[0], model_size, context_length, series=series, **kwargs)
        else:
            # Multiple specific models requested
            return _forecast_specific_models(csv_path, target_col, horizon, forecaster, model_size, context_length, series=series, **kwargs)

    # If specific forecaster requested (not multi), use single model
    if forecaster != "multi" and forecaster.lower() != "ensemble":
        return _forecast_single(csv_path, target_col, horizon, forecaster, model_size, context_length, series=series, **kwargs)

    # Multi-model forecasting (uses all available)
    return _forecast_multi_model(csv_path, target_col, horizon, model_size, context_length, series=series, **kwargs)


def _forecast_single(
    csv_path: str,
    target_col: str,
    horizon: int,
    forecaster: str,
    model_size: str,
    context_length: int,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """Run forecast with a single model."""
    # Pop provider from kwargs - only LLM forecaster uses it
    provider = kwargs.pop("provider", None)

    logger.info(f"Running forecast: {forecaster} on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 3:
            return {
                "success": False,
                "error": "Need at least 3 numeric data points for forecasting"
            }

        # Models that accept model_size parameter
        MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}

        try:
            if forecaster.lower() == "llm":
                if not provider:
                    return {"success": False, "error": "LLM forecaster requires 'provider'"}
                fc = create_forecaster("llm", provider=provider)
            elif forecaster.lower() in MODELS_WITH_SIZE:
                fc = create_forecaster(forecaster, model_size=model_size, **kwargs)
            else:
                fc = create_forecaster(forecaster, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create forecaster: {e}")
            return {"success": False, "error": f"Failed to create forecaster: {e}"}

        result = fc.forecast(series=series, horizon=horizon, context_length=context_length)

        if result.success:
            return _to_python_types({
                "success": True,
                "forecast": result.forecast,
                "timestamps": result.timestamps,
                "uncertainty": result.uncertainty,
                "horizon": result.horizon,
                "model": result.model_name,
                "models_used": [result.model_name or forecaster],
                "metadata": result.metadata
            })
        else:
            return {"success": False, "error": result.error}

    except Exception as e:
        import traceback
        logger.error(f"Forecast error: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _forecast_multi_model(
    csv_path: str,
    target_col: str,
    horizon: int,
    model_size: str,
    context_length: int,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """Run forecast with multiple models and aggregate results."""
    logger.info(f"Multi-model forecasting on {csv_path or 'series'}:{target_col or 'provided'}")

    # Extract LLM provider for intelligent model selection (injected by orchestrator)
    provider = kwargs.pop("provider", None)
    selection_method = kwargs.pop("selection_method", "auto")

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 3:
            return {"success": False, "error": "Need at least 3 numeric data points"}

        # Cross-validation-based model selection
        use_cv = (
            selection_method == "cv"
            or (selection_method == "auto" and len(series) > 100)
        )

        if use_cv:
            try:
                from ..multi_model.cross_validation import select_model_with_cv
                cv_result = select_model_with_cv(
                    series=series,
                    horizon=horizon,
                    provider=provider,
                    model_size=model_size,
                )
                return _build_cv_forecast_response(cv_result, horizon, series)
            except Exception as e:
                logger.warning(f"CV selection failed ({e}), falling back to static selection")

        # Standard selection: 3-5 models based on available forecasters
        available = get_available_forecasters()
        models_to_use = _select_forecast_models(series, available, provider=provider)

        logger.info(f"Using {len(models_to_use)} forecasters: {models_to_use}")

        # Run each forecaster
        all_forecasts = []
        model_details = []
        models_succeeded = []
        model_results = {}  # Track individual model results for frontend chart
        timestamps = None

        # Models that accept model_size parameter
        MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}

        for model_name in models_to_use:
            try:
                # Only pass model_size to forecasters that support it
                if model_name in MODELS_WITH_SIZE:
                    fc = create_forecaster(model_name, model_size=model_size)
                else:
                    fc = create_forecaster(model_name)

                if fc is None:
                    continue

                result = fc.forecast(series=series, horizon=horizon, context_length=context_length)

                if result.success:
                    models_succeeded.append(model_name)
                    all_forecasts.append(result.forecast)

                    if timestamps is None:
                        timestamps = result.timestamps

                    model_details.append({
                        "model": model_name,
                        "forecast": result.forecast,
                        "rationale": MODEL_RATIONALE.get(model_name, f"{model_name}: Specialized forecaster")
                    })

                    # Track for frontend multi_model chart display
                    model_results[model_name] = {
                        "success": True,
                        "data": {
                            "predictions": result.forecast,
                            "timestamps": result.timestamps,
                            "model_name": result.model_name or model_name
                        }
                    }
            except Exception as e:
                logger.warning(f"Forecaster {model_name} failed: {e}")
                continue

        if not all_forecasts:
            return {"success": False, "error": "All forecasters failed"}

        # Aggregate forecasts - compute ensemble mean and uncertainty
        forecast_array = np.array(all_forecasts)
        ensemble_mean = np.mean(forecast_array, axis=0).tolist()
        ensemble_std = np.std(forecast_array, axis=0).tolist()

        # Calculate prediction spread (disagreement between models)
        prediction_spread = np.mean(ensemble_std)

        # Build model selection explanation
        model_selection_rationale = _build_model_rationale(series, models_succeeded)

        return _to_python_types({
            "success": True,
            "forecast": ensemble_mean,
            "timestamps": timestamps,
            "uncertainty": ensemble_std,
            "horizon": horizon,
            "model": "Multi-Model Ensemble",
            "models_used": models_succeeded,
            "model_details": model_details,
            "model_results": model_results,  # Individual model predictions for chart
            "model_selection_rationale": model_selection_rationale,
            "comparison": {
                "num_models": len(models_succeeded),
                "prediction_spread": round(float(prediction_spread), 2),
                "forecast_range": {
                    "min": [round(float(min(forecast_array[:, i])), 2) for i in range(len(ensemble_mean))],
                    "max": [round(float(max(forecast_array[:, i])), 2) for i in range(len(ensemble_mean))],
                }
            },
            "metadata": {
                "ensemble_method": "mean",
                "individual_forecasts": {m["model"]: m["forecast"] for m in model_details}
            }
        })

    except Exception as e:
        import traceback
        logger.error(f"Multi-model forecast error: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _forecast_specific_models(
    csv_path: str,
    target_col: str,
    horizon: int,
    models: List[str],
    model_size: str,
    context_length: int,
    series: pd.Series = None,
    **kwargs
) -> Dict[str, Any]:
    """Run forecast with specific user-requested models and return combined results."""
    # Pop provider from kwargs - only LLM forecaster uses it
    provider = kwargs.pop("provider", None)

    logger.info(f"Running specific models: {models} on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 3:
            return {"success": False, "error": "Need at least 3 numeric data points"}

        # Models that accept model_size parameter
        MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}

        # Run each specified forecaster
        all_forecasts = []
        model_details = []
        models_succeeded = []
        timestamps = None
        model_results = {}  # For multi_model chart format

        for model_name in models:
            try:
                if model_name in MODELS_WITH_SIZE:
                    fc = create_forecaster(model_name, model_size=model_size)
                else:
                    fc = create_forecaster(model_name)

                if fc is None:
                    logger.warning(f"Could not create forecaster: {model_name}")
                    continue

                result = fc.forecast(series=series, horizon=horizon, context_length=context_length)

                if result.success:
                    models_succeeded.append(model_name)
                    all_forecasts.append(result.forecast)

                    if timestamps is None:
                        timestamps = result.timestamps

                    model_details.append({
                        "model": model_name,
                        "forecast": result.forecast,
                        "rationale": MODEL_RATIONALE.get(model_name, f"{model_name}: Specialized forecaster")
                    })

                    # Format for frontend multi_model chart
                    model_results[model_name] = {
                        "success": True,
                        "data": {
                            "predictions": result.forecast,
                            "timestamps": result.timestamps,
                            "model_name": result.model_name
                        }
                    }
            except Exception as e:
                logger.warning(f"Forecaster {model_name} failed: {e}")
                continue

        if not all_forecasts:
            return {"success": False, "error": f"All specified forecasters failed: {models}"}

        # Aggregate forecasts
        forecast_array = np.array(all_forecasts)
        ensemble_mean = np.mean(forecast_array, axis=0).tolist()
        ensemble_std = np.std(forecast_array, axis=0).tolist()
        prediction_spread = np.mean(ensemble_std)

        # Build rationale
        model_selection_rationale = _build_model_rationale(series, models_succeeded)

        return _to_python_types({
            "success": True,
            "forecast": ensemble_mean,
            "timestamps": timestamps,
            "uncertainty": ensemble_std,
            "horizon": horizon,
            "model": f"Ensemble ({', '.join(models_succeeded)})",
            "models_used": models_succeeded,
            "model_details": model_details,
            "model_results": model_results,  # For multi_model chart
            "model_selection_rationale": model_selection_rationale,
            "comparison": {
                "num_models": len(models_succeeded),
                "prediction_spread": round(float(prediction_spread), 2),
                "forecast_range": {
                    "min": [round(float(min(forecast_array[:, i])), 2) for i in range(len(ensemble_mean))],
                    "max": [round(float(max(forecast_array[:, i])), 2) for i in range(len(ensemble_mean))],
                }
            },
            "metadata": {
                "ensemble_method": "mean",
                "requested_models": models,
                "individual_forecasts": {m["model"]: m["forecast"] for m in model_details}
            }
        })

    except Exception as e:
        import traceback
        logger.error(f"Specific models forecast error: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _select_forecast_models(series: pd.Series, available: List[str], provider=None) -> List[str]:
    """Select 3-5 appropriate forecast models based on data characteristics.

    If an LLM provider is given, asks the LLM to choose models based on the
    series statistical profile. Falls back to the static priority list when
    no provider is available or the LLM call fails.
    """
    # Try LLM-driven selection first
    if provider is not None:
        from .model_selector import llm_select_models
        llm_selected = llm_select_models(
            series=series,
            available=[m for m in available if m != "llm"],
            model_descriptions=MODEL_RATIONALE,
            provider=provider,
            task_type="forecasting",
        )
        if llm_selected:
            return llm_selected

    # Static fallback
    selected = []

    # Always include baseline (no dependencies)
    if "zscore_baseline" in available:
        selected.append("zscore_baseline")
    elif "baseline" in available:
        selected.append("baseline")

    # Prefer foundation models if available
    priority_models = ["moirai", "chronos", "timesfm", "tirex", "lag-llama"]

    for model in priority_models:
        if model in available and len(selected) < 5:
            selected.append(model)

    # Ensure we have at least 3 models
    for model in available:
        if model not in selected and model != "llm" and len(selected) < 3:
            selected.append(model)

    return selected[:5]  # Cap at 5


def _build_model_rationale(series: pd.Series, models_used: List[str]) -> str:
    """Build explanation for why these models were selected."""
    rationale_parts = [
        f"Selected {len(models_used)} complementary forecasting methods for robust predictions:\n"
    ]

    for model in models_used:
        if model in MODEL_RATIONALE:
            rationale_parts.append(f"• {MODEL_RATIONALE[model]}")
        else:
            rationale_parts.append(f"• {model}: Specialized time series forecaster")

    rationale_parts.append(
        f"\nEnsemble approach: Final forecast is the mean of all models, "
        f"uncertainty reflects model disagreement."
    )

    return "\n".join(rationale_parts)


def _build_cv_forecast_response(cv_result, horizon: int, series: pd.Series) -> Dict[str, Any]:
    """Build forecast response from cross-validation selection result."""
    successful = {k: v for k, v in cv_result.cv_results.items() if v.success}

    if not successful:
        return {"success": False, "error": "All models failed during cross-validation"}

    selected = cv_result.selected_model
    models_succeeded = list(successful.keys())

    if selected == "mixture" or selected not in successful:
        # Mixture: average all model forecasts
        all_fwd = [np.array(r.forecast) for r in successful.values() if r.forecast]
        if not all_fwd:
            return {"success": False, "error": "No valid forecasts from CV models"}
        min_len = min(len(f) for f in all_fwd)
        all_fwd = [f[:min_len] for f in all_fwd]
        forecast_array = np.array(all_fwd)
        ensemble_mean = np.mean(forecast_array, axis=0).tolist()
        ensemble_std = np.std(forecast_array, axis=0).tolist()
        model_label = "CV Mixture Ensemble"
    else:
        # Single best model
        best = successful[selected]
        ensemble_mean = best.forecast
        ensemble_std = [0.0] * len(best.forecast)
        model_label = f"CV-Selected: {selected}"

    # Generate timestamps
    from ..utils import infer_frequency
    freq = infer_frequency(series)
    future_index = pd.date_range(start=series.index[-1], periods=len(ensemble_mean) + 1, freq=freq)[1:]
    timestamps = [str(t) for t in future_index]

    # Build model details
    model_details = []
    model_results = {}
    for name, r in successful.items():
        model_details.append({
            "model": name,
            "forecast": r.forecast,
            "cv_mae": round(r.mae, 4),
            "rationale": MODEL_RATIONALE.get(name, f"{name}: Specialized forecaster"),
        })
        model_results[name] = {
            "success": True,
            "data": {
                "predictions": r.forecast,
                "timestamps": timestamps[:len(r.forecast)],
                "model_name": name,
            }
        }

    return _to_python_types({
        "success": True,
        "forecast": ensemble_mean,
        "timestamps": timestamps[:len(ensemble_mean)],
        "uncertainty": ensemble_std,
        "horizon": horizon,
        "model": model_label,
        "models_used": models_succeeded,
        "model_details": model_details,
        "model_results": model_results,
        "model_selection_rationale": (
            f"Cross-validation-based selection.\n"
            f"Selected: {selected} ({cv_result.selection_method}).\n"
            f"MAE ranking: {', '.join(f'{m}={e:.4f}' for m, e in cv_result.mae_ranking[:5])}\n"
            f"Diversity: {'diverse' if cv_result.is_diverse else 'low'} "
            f"(score={cv_result.diversity_score:.4f})"
        ),
        "cv_selection": cv_result.to_dict(),
        "comparison": {
            "num_models": len(models_succeeded),
            "prediction_spread": round(float(np.mean(ensemble_std)) if ensemble_std else 0, 2),
            "is_diverse": cv_result.is_diverse,
            "diversity_score": round(cv_result.diversity_score, 4),
        },
        "metadata": {
            "ensemble_method": "cv_selection",
            "selection_method": cv_result.selection_method,
            "selected_model": cv_result.selected_model,
        }
    })


# Tool registration info
TOOL_INFO = {
    "name": "run_forecast",
    "function": run_forecast,
    "description": "Generate forecasts using multiple ML models (3-5 forecasters) for robust predictions",
    "parameters": {
        "csv_path": "Path to CSV file with time series",
        "target_col": "Name of column to forecast",
        "horizon": "Number of steps to predict (default 7)",
        "forecaster": f"'multi' for ensemble (default), or specific: {', '.join(get_available_forecasters())}",
        "model_size": "Model size: small, base, large (default small)"
    }
}
