# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Quantile-aware ensemble blending for Time Series Council.

Instead of averaging point forecasts, this module:
1. Collects quantile predictions (0.1 through 0.9) from each model
2. For "mixture" blending: concatenates all quantile samples, re-computes quantiles
3. For single-model selection: shifts mixture quantiles to align with
   the selected model's median
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

from ..types import ForecastResult
from ..logging import get_logger

logger = get_logger(__name__)

STANDARD_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class QuantileEnsembleResult:
    """Result from quantile-aware ensemble."""
    success: bool
    median_forecast: List[float] = field(default_factory=list)
    quantile_forecasts: Dict[str, List[float]] = field(default_factory=dict)
    selected_model: Optional[str] = None
    blend_method: str = "mixture"  # "mixture" or "selected_shifted"
    model_quantiles: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "median_forecast": self.median_forecast,
            "quantile_forecasts": self.quantile_forecasts,
            "selected_model": self.selected_model,
            "blend_method": self.blend_method,
        }


def collect_quantile_predictions(
    model_results: Dict[str, ForecastResult],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract quantile predictions from model results.

    For models that produce quantiles (in metadata['quantiles']):
        use their native quantile outputs.
    For models that only produce point + uncertainty:
        approximate quantiles assuming Gaussian distribution.

    Returns:
        Dict[model_name, Dict[quantile_str, List[float]]]
    """
    all_quantiles = {}

    for model_name, result in model_results.items():
        if not result.success or not result.forecast:
            continue

        # Check for native quantiles in metadata
        native_quantiles = None
        if result.metadata and "quantiles" in result.metadata:
            native_quantiles = result.metadata["quantiles"]

        if native_quantiles and len(native_quantiles) >= 5:
            all_quantiles[model_name] = native_quantiles
        else:
            # Approximate from point forecast + uncertainty (assuming Gaussian)
            forecast = np.array(result.forecast, dtype=float)
            uncertainty = np.array(result.uncertainty, dtype=float) if result.uncertainty else np.ones_like(forecast)

            model_q = {}
            for q in STANDARD_QUANTILES:
                z = stats.norm.ppf(q)
                model_q[str(q)] = (forecast + z * uncertainty).tolist()
            all_quantiles[model_name] = model_q

    return all_quantiles


def mixture_blend(
    model_quantiles: Dict[str, Dict[str, List[float]]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    """
    Mixture blending: pool quantile samples from all models, re-quantile.

    For each time step and each quantile level:
    1. Gather values from all models at all quantile levels
    2. Re-compute the target quantile from the pooled set
    """
    if not model_quantiles:
        return {}

    # Determine forecast length
    first_model = next(iter(model_quantiles.values()))
    first_key = next(iter(first_model.keys()))
    n_steps = len(first_model[first_key])

    # Pool all quantile values across models
    result = {}
    for q_level in STANDARD_QUANTILES:
        q_str = str(q_level)
        blended = []

        for step in range(n_steps):
            pooled = []
            for model_name, q_dict in model_quantiles.items():
                w = weights.get(model_name, 1.0) if weights else 1.0
                # Collect all quantile values at this step from this model
                for q_key, q_vals in q_dict.items():
                    if step < len(q_vals):
                        pooled.append(q_vals[step])

            if pooled:
                blended.append(float(np.quantile(pooled, q_level)))
            else:
                blended.append(0.0)

        result[q_str] = blended

    return result


def shift_to_selected(
    mixture_quantiles: Dict[str, List[float]],
    selected_model_quantiles: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """
    Shift mixture quantiles to align with selected model's median.

    Computes offset = selected_median - mixture_median at each time step,
    then shifts all mixture quantiles by that offset.
    """
    mixture_median = np.array(mixture_quantiles.get("0.5", []))
    selected_median = np.array(selected_model_quantiles.get("0.5", []))

    if len(mixture_median) == 0 or len(selected_median) == 0:
        return mixture_quantiles

    min_len = min(len(mixture_median), len(selected_median))
    offset = selected_median[:min_len] - mixture_median[:min_len]

    shifted = {}
    for q_str, q_vals in mixture_quantiles.items():
        vals = np.array(q_vals[:min_len])
        shifted[q_str] = (vals + offset).tolist()

    return shifted


def quantile_ensemble(
    model_results: Dict[str, ForecastResult],
    selected_model: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> QuantileEnsembleResult:
    """
    Main entry point for quantile-aware ensemble blending.

    If selected_model is None or "mixture": pure mixture blend.
    If selected_model is a specific model: shift mixture to that model's median.
    """
    # Collect quantile predictions from all models
    model_quantiles = collect_quantile_predictions(model_results)

    if not model_quantiles:
        return QuantileEnsembleResult(success=False)

    # Compute mixture blend
    blended = mixture_blend(model_quantiles, weights)

    if not blended:
        return QuantileEnsembleResult(success=False)

    # Determine blend method and apply shift if needed
    if selected_model and selected_model != "mixture" and selected_model in model_quantiles:
        final_quantiles = shift_to_selected(blended, model_quantiles[selected_model])
        blend_method = "selected_shifted"
    else:
        final_quantiles = blended
        blend_method = "mixture"

    median = final_quantiles.get("0.5", [])

    return QuantileEnsembleResult(
        success=True,
        median_forecast=median,
        quantile_forecasts=final_quantiles,
        selected_model=selected_model,
        blend_method=blend_method,
        model_quantiles=model_quantiles,
    )
