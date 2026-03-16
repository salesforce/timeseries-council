# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Simulation tools for the orchestrator.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)


def what_if_simulation(
    csv_path: str = None,
    target_col: str = None,
    scale_factor: float = 1.2,
    horizon: int = 14,
    apply_to_last: int = None,
    series: pd.Series = None,
) -> Dict[str, Any]:
    """
    Scenario analysis - simulate impact of scaling the data.

    Args:
        csv_path: Path to CSV file
        target_col: Column to analyze
        scale_factor: Multiplier for simulation (e.g., 1.2 = 20% increase)
        horizon: Forecast horizon for comparison
        apply_to_last: If set, only apply scaling to the last N data points
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)

    Returns:
        Dict with baseline vs simulated scenario comparison
    """
    logger.info(f"Running what-if simulation: {scale_factor}x on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        from ._utils import prepare_series

        series = prepare_series(csv_path=csv_path, target_col=target_col, series=series)

        if len(series) < 2:
            return {
                "success": False,
                "error": "Need at least 2 data points for simulation"
            }

        # Calculate modified series
        original_series = series.copy()
        modified_series = series.copy()
        
        modification_start_idx = 0
        
        if apply_to_last and apply_to_last > 0:
            # Apply only to last N points
            if apply_to_last >= len(series):
                # Apply to all if N is greater than length
                modified_series = modified_series * scale_factor
            else:
                # Apply only to last N
                modification_start_idx = len(series) - apply_to_last
                modified_series.iloc[-apply_to_last:] = modified_series.iloc[-apply_to_last:] * scale_factor
                logger.info(f"applied scaling only to last {apply_to_last} points")
        else:
            # Apply to whole series
            modified_series = modified_series * scale_factor

        # Current state
        current_mean = float(original_series.mean())
        current_latest = float(original_series.iloc[-1])

        # Simulated state
        simulated_mean = float(modified_series.mean())
        simulated_latest = float(modified_series.iloc[-1])

        # Simple projection (using recent trend of the MODIFIED series)
        if len(modified_series) >= 7:
            recent_trend = (modified_series.iloc[-1] - modified_series.iloc[-7]) / 7
            baseline_projection = current_latest + (recent_trend * horizon) # Note: this uses modified trend? Or should use original?
            # Usually what-if assumes the change persists. 
            # But let's keep baseline projection based on ORIGINAL trend for fair comparison?
            
            # Recalculate original trend for baseline
            orig_trend = (original_series.iloc[-1] - original_series.iloc[-7]) / 7
            baseline_projection = current_latest + (orig_trend * horizon)
            
            simulated_projection = simulated_latest + (recent_trend * horizon)
        else:
            baseline_projection = current_latest
            simulated_projection = simulated_latest

        # Impact analysis
        absolute_impact = simulated_projection - baseline_projection
        percent_impact = ((simulated_projection - baseline_projection) / abs(baseline_projection)) * 100 if baseline_projection != 0 else 0

        # Prepare chart data
        # We need timestamps (indexes) converted to string
        timestamps = series.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        
        chart_data = {
            "timestamps": timestamps,
            "original_values": original_series.tolist(),
            "simulated_values": modified_series.tolist(),
            "modification_start_index": modification_start_idx
        }

        logger.info(f"Simulation complete: {percent_impact:.1f}% impact")

        return {
            "success": True,
            "scenario": f"{(scale_factor - 1) * 100:+.0f}% change{' (last ' + str(apply_to_last) + ' points)' if apply_to_last else ''}",
            "scale_factor": scale_factor,
            "horizon": horizon,
            "baseline": {
                "current_mean": round(current_mean, 2),
                "current_latest": round(current_latest, 2),
                "projected_value": round(float(baseline_projection), 2)
            },
            "simulated": {
                "adjusted_mean": round(simulated_mean, 2),
                "adjusted_latest": round(simulated_latest, 2),
                "projected_value": round(float(simulated_projection), 2)
            },
            "impact": {
                "absolute_difference": round(float(absolute_impact), 2),
                "percent_difference": round(float(percent_impact), 2)
            },
            "chart_data": chart_data
        }

    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def sensitivity_analysis(
    csv_path: str = None,
    target_col: str = None,
    parameter: str = "scale",
    values: list = None,
    horizon: int = 14,
    series: pd.Series = None,
) -> Dict[str, Any]:
    """
    Run multiple what-if scenarios to understand sensitivity.

    Args:
        csv_path: Path to CSV file
        target_col: Column to analyze
        parameter: Parameter to vary ("scale" or "horizon")
        values: List of values to test
        horizon: Base forecast horizon
        series: Pre-loaded pd.Series with DatetimeIndex (alternative to csv_path+target_col)

    Returns:
        Dict with sensitivity analysis results
    """
    logger.info(f"Running sensitivity analysis: {parameter} on {csv_path or 'series'}:{target_col or 'provided'}")

    try:
        if values is None:
            if parameter == "scale":
                values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
            else:
                values = [7, 14, 21, 30]

        results = []

        for val in values:
            if parameter == "scale":
                sim = what_if_simulation(csv_path, target_col, scale_factor=val, horizon=horizon, series=series)
            else:
                sim = what_if_simulation(csv_path, target_col, scale_factor=1.0, horizon=int(val), series=series)

            if sim["success"]:
                results.append({
                    "parameter_value": val,
                    "projected_value": sim["simulated"]["projected_value"],
                    "impact_pct": sim["impact"]["percent_difference"]
                })

        if not results:
            return {
                "success": False,
                "error": "All simulations failed"
            }

        # Calculate sensitivity metrics
        projections = [r["projected_value"] for r in results]
        impacts = [r["impact_pct"] for r in results]

        logger.info(f"Sensitivity analysis complete: {len(results)} scenarios")

        return {
            "success": True,
            "parameter": parameter,
            "scenarios": results,
            "summary": {
                "min_projection": min(projections),
                "max_projection": max(projections),
                "range": max(projections) - min(projections),
                "avg_impact": np.mean(impacts) if impacts else 0
            }
        }

    except Exception as e:
        logger.error(f"Sensitivity analysis error: {e}")
        return {"success": False, "error": str(e)}


# Tool registration info
WHAT_IF_TOOL = {
    "name": "what_if_simulation",
    "function": what_if_simulation,
    "description": "Scenario analysis - simulate impact of scaling the data",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to analyze",
        "scale_factor": "Multiplier (e.g., 1.2 = +20%, 0.8 = -20%)",
        "horizon": "Forecast horizon for projection (default 14)",
        "apply_to_last": "Optional: Only apply to last N data points"
    }
}

SENSITIVITY_TOOL = {
    "name": "sensitivity_analysis",
    "function": sensitivity_analysis,
    "description": "Run multiple scenarios to understand parameter sensitivity",
    "parameters": {
        "csv_path": "Path to CSV file",
        "target_col": "Name of column to analyze",
        "parameter": "Parameter to vary: 'scale' or 'horizon'",
        "values": "Optional list of values to test"
    }
}
