# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Context-aware forecasting tool for Time Series Council.

Accepts textual context (e.g., "heat wave expected", "holiday season") and uses
LLM reasoning to adjust forecasts through three mechanisms:
  a. Lookback window selection (trim to relevant history)
  b. Anomaly refinement (remove non-persistent outliers)
  c. Future effects anticipation (adjust for described events)
"""

import json
import re
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

from ..logging import get_logger
from .forecasting import run_forecast, _to_python_types

logger = get_logger(__name__)

MAX_REASONING_ITERATIONS = 6

CONTEXT_REASONING_PROMPT = """\
You are solving a contextual time-series forecasting problem where historical \
values and contextual information are provided.

HISTORICAL DATA SUMMARY:
- Length: {length} points
- Date range: {date_start} to {date_end}
- Mean: {mean:.2f}, Std: {std:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}
- Recent trend: {trend}
- Last 10 values: {recent_values}

CONTEXT INFORMATION:
{context_info}

BASELINE FORECAST (next {horizon} steps):
{baseline_forecast}

Based on the context, determine if the baseline forecast should be adjusted. \
Think carefully about whether the context dominates the historical pattern, \
whether any historical anomalies are non-persistent, and whether upcoming \
events should change the forecast.

Respond with EXACTLY ONE JSON object (no markdown code blocks), choosing one action:

1. Trim lookback (regime change detected):
{{"action": "trim_lookback", "keep_last_n_points": <int>, "reason": "..."}}

2. Remove anomalies (non-persistent outliers):
{{"action": "remove_anomalies", "timestamps_to_remove": ["YYYY-MM-DD", ...], "reason": "..."}}

3. Adjust forecast (upcoming events):
{{"action": "adjust_forecast", "adjustments": [{{"step": <0-based index>, "factor": <multiplier>}}, ...], "reason": "..."}}

4. No adjustment needed:
{{"action": "none", "reason": "..."}}
"""


def context_forecast(
    csv_path: str,
    target_col: str,
    context_info: str,
    horizon: int = 7,
    forecaster: str = "multi",
    model_size: str = "small",
    max_iterations: int = MAX_REASONING_ITERATIONS,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run context-aware forecast with LLM reasoning.

    Args:
        csv_path: Path to CSV with time series
        target_col: Column to forecast
        context_info: Textual context (e.g., "heat wave expected next week")
        horizon: Forecast horizon
        forecaster: Forecaster to use
        model_size: Model size
        max_iterations: Max LLM reasoning iterations
        **kwargs: Additional arguments (provider injected by orchestrator)

    Returns:
        Dict with baseline forecast, adjusted forecast, reasoning chain,
        and applied adjustments.
    """
    provider = kwargs.pop("provider", None)

    if not provider:
        return {
            "success": False,
            "error": "Context-aware forecasting requires an LLM provider"
        }

    if not context_info or not context_info.strip():
        return {
            "success": False,
            "error": "context_info is required for context-aware forecasting"
        }

    try:
        from ..utils import load_timeseries_csv
        df = load_timeseries_csv(csv_path)

        if target_col not in df.columns:
            return {
                "success": False,
                "error": f"Column '{target_col}' not found. Available: {list(df.columns)}"
            }

        series = df[target_col].dropna()
        series = pd.to_numeric(series, errors='coerce').dropna()

        if len(series) < 3:
            return {"success": False, "error": "Need at least 3 numeric data points"}

        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        # Step 1: Get baseline forecast
        baseline_result = run_forecast(
            csv_path=csv_path,
            target_col=target_col,
            horizon=horizon,
            forecaster=forecaster,
            model_size=model_size,
            selection_method="static",
            provider=provider,
        )

        if not baseline_result.get("success"):
            return baseline_result

        baseline_forecast = baseline_result["forecast"]
        reasoning_chain = []
        adjustments_applied = []
        current_series = series.copy()
        current_forecast = list(baseline_forecast)

        # Step 2: LLM reasoning loop
        for iteration in range(min(max_iterations, MAX_REASONING_ITERATIONS)):
            prompt = _build_context_prompt(
                current_series, context_info, current_forecast, horizon
            )

            try:
                response = provider.generate(prompt)
                action = _parse_context_action(response)
            except Exception as e:
                logger.warning(f"LLM reasoning iteration {iteration} failed: {e}")
                reasoning_chain.append({
                    "iteration": iteration,
                    "error": str(e),
                })
                break

            reasoning_chain.append({
                "iteration": iteration,
                "action": action.get("action", "none"),
                "reason": action.get("reason", ""),
            })

            action_type = action.get("action", "none")

            if action_type == "trim_lookback":
                keep_n = action.get("keep_last_n_points", len(current_series))
                keep_n = max(20, min(keep_n, len(current_series)))
                current_series = current_series.tail(keep_n)
                adjustments_applied.append({
                    "type": "trim_lookback",
                    "keep_last_n": keep_n,
                    "reason": action.get("reason", ""),
                })
                # Re-forecast with trimmed series
                trimmed_result = run_forecast(
                    csv_path=csv_path,
                    target_col=target_col,
                    horizon=horizon,
                    forecaster=forecaster,
                    model_size=model_size,
                    selection_method="static",
                    context_length=keep_n,
                    provider=provider,
                )
                if trimmed_result.get("success"):
                    current_forecast = list(trimmed_result["forecast"])

            elif action_type == "remove_anomalies":
                timestamps = action.get("timestamps_to_remove", [])
                if timestamps:
                    current_series = _apply_anomaly_removal(current_series, timestamps)
                    adjustments_applied.append({
                        "type": "remove_anomalies",
                        "removed": timestamps,
                        "reason": action.get("reason", ""),
                    })

            elif action_type == "adjust_forecast":
                adj_list = action.get("adjustments", [])
                if adj_list:
                    current_forecast = _apply_forecast_adjustment(current_forecast, adj_list)
                    adjustments_applied.append({
                        "type": "adjust_forecast",
                        "adjustments": adj_list,
                        "reason": action.get("reason", ""),
                    })
                break  # Forecast adjustment is the final step

            elif action_type == "none":
                break

        return _to_python_types({
            "success": True,
            "forecast": current_forecast,
            "baseline_forecast": baseline_forecast,
            "timestamps": baseline_result.get("timestamps", []),
            "uncertainty": baseline_result.get("uncertainty", []),
            "horizon": horizon,
            "model": baseline_result.get("model", ""),
            "models_used": baseline_result.get("models_used", []),
            "context_info": context_info,
            "context_adjustments": adjustments_applied,
            "reasoning_chain": reasoning_chain,
            "forecast_changed": current_forecast != list(baseline_forecast),
            "metadata": {
                "context_aware": True,
                "iterations_used": len(reasoning_chain),
                "baseline_model": baseline_result.get("model", ""),
            }
        })

    except Exception as e:
        import traceback
        logger.error(f"Context forecast error: {e}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def _build_context_prompt(
    series: pd.Series,
    context_info: str,
    current_forecast: list,
    horizon: int,
) -> str:
    """Build the LLM prompt with series data and context."""
    values = series.values.astype(float)
    recent = values[-10:] if len(values) >= 10 else values

    # Simple trend detection
    if len(values) >= 7:
        recent_mean = float(np.mean(values[-7:]))
        older_mean = float(np.mean(values[-14:-7])) if len(values) >= 14 else float(np.mean(values))
        if recent_mean > older_mean * 1.05:
            trend = "upward"
        elif recent_mean < older_mean * 0.95:
            trend = "downward"
        else:
            trend = "stable"
    else:
        trend = "insufficient data"

    forecast_str = ", ".join([f"{v:.2f}" for v in current_forecast[:20]])
    if len(current_forecast) > 20:
        forecast_str += f" ... ({len(current_forecast)} total)"

    return CONTEXT_REASONING_PROMPT.format(
        length=len(series),
        date_start=str(series.index[0]),
        date_end=str(series.index[-1]),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        trend=trend,
        recent_values=", ".join([f"{v:.2f}" for v in recent]),
        context_info=context_info,
        horizon=horizon,
        baseline_forecast=forecast_str,
    )


def _apply_anomaly_removal(
    series: pd.Series,
    timestamps_to_remove: List[str],
) -> pd.Series:
    """Remove specified timestamps and interpolate."""
    result = series.copy()
    for ts_str in timestamps_to_remove:
        try:
            ts = pd.Timestamp(ts_str)
            # Find nearest index
            idx = result.index.get_indexer([ts], method="nearest")[0]
            if 0 <= idx < len(result):
                result.iloc[idx] = np.nan
        except Exception:
            continue

    # Interpolate removed points
    result = result.interpolate(method="linear")
    result = result.ffill().bfill()
    return result


def _apply_forecast_adjustment(
    forecast_values: List[float],
    adjustments: List[Dict[str, Any]],
) -> List[float]:
    """Apply multiplicative adjustments to forecast."""
    result = list(forecast_values)
    for adj in adjustments:
        step = adj.get("step", 0)
        factor = adj.get("factor", 1.0)
        if 0 <= step < len(result):
            result[step] = result[step] * factor
    return result


def _parse_context_action(response: str) -> Dict[str, Any]:
    """Parse LLM response for context action JSON."""
    # Try to find JSON in the response
    # First, try raw JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON within markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object in the response
    brace_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: no adjustment
    return {"action": "none", "reason": "Could not parse LLM response"}


# Tool registration info
TOOL_INFO = {
    "name": "context_forecast",
    "function": context_forecast,
    "description": (
        "Context-aware forecasting: provide textual context (e.g., 'heat wave next week', "
        "'holiday season', 'new product launch') to adjust forecasts based on domain knowledge. "
        "The LLM reasons about how context affects the forecast."
    ),
    "parameters": {
        "csv_path": "Path to CSV file with time series",
        "target_col": "Name of column to forecast",
        "context_info": "Textual description of context (e.g., 'heat wave expected next week')",
        "horizon": "Number of steps to predict (default 7)",
        "forecaster": "'multi' for ensemble (default), or specific forecaster name",
        "model_size": "Model size: small, base, large (default small)",
    }
}
