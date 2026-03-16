# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Cross-validation-based model selection for Time Series Council.

Runs all candidate models on a held-out validation set (last N points),
computes MAE for each, normalizes series data, and optionally uses an LLM
to select the best model based on actual performance evidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

from ..forecasters import create_forecaster, get_available_forecasters
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrossValResult:
    """Result from cross-validation for a single model."""
    model_name: str
    mae: float
    cv_predictions: List[float]  # Predictions on held-out validation set
    forecast: List[float]  # Full-horizon forward predictions
    quantiles: Optional[Dict[str, List[float]]] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class CVSelectionResult:
    """Result of cross-validation-based model selection."""
    selected_model: str
    selection_method: str  # "llm", "best_mae", "mixture"
    cv_results: Dict[str, CrossValResult] = field(default_factory=dict)
    mae_ranking: List[Tuple[str, float]] = field(default_factory=list)
    diversity_score: float = 0.0
    is_diverse: bool = True
    llm_reasoning: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "selected_model": self.selected_model,
            "selection_method": self.selection_method,
            "mae_ranking": [{"model": m, "mae": round(e, 4)} for m, e in self.mae_ranking],
            "diversity_score": round(self.diversity_score, 4),
            "is_diverse": self.is_diverse,
            "llm_reasoning": self.llm_reasoning,
        }


def normalize_series(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Z-score normalize: subtract mean, divide by std."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return values, 0.0, 1.0
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    if std < 1e-10:
        std = 1.0
    normalized = (values - mean) / std
    return normalized, mean, std


def downsample_for_llm(values: np.ndarray, max_points: int = 40) -> np.ndarray:
    """Downsample long series to max_points for LLM consumption."""
    if len(values) <= max_points:
        return values
    step = max(1, len(values) // max_points)
    return values[::step][:max_points]


def check_prediction_diversity(
    forecasts: Dict[str, List[float]],
    threshold: float = 0.01,
) -> Tuple[bool, float]:
    """
    Check if model predictions are diverse enough to warrant LLM selection.

    If std across model median forecasts < threshold after normalization,
    predictions are too similar and mixture should be used directly.

    Returns:
        (is_diverse, diversity_score)
    """
    if len(forecasts) < 2:
        return True, 1.0

    predictions = []
    for vals in forecasts.values():
        predictions.append(np.array(vals, dtype=float))

    # Align lengths
    min_len = min(len(p) for p in predictions)
    predictions = [p[:min_len] for p in predictions]
    pred_array = np.array(predictions)

    # Normalize before comparing
    mean = np.mean(pred_array)
    std = np.std(pred_array)
    if std < 1e-10:
        return False, 0.0

    normalized = (pred_array - mean) / std
    diversity_score = float(np.mean(np.std(normalized, axis=0)))
    is_diverse = diversity_score > threshold

    return is_diverse, diversity_score


def run_cross_validation(
    series: pd.Series,
    horizon: int,
    cv_points: Optional[int] = None,
    model_names: Optional[List[str]] = None,
    model_size: str = "small",
) -> Dict[str, CrossValResult]:
    """
    Run cross-validation on all candidate models.

    Splits series into:
      - training: series[:-cv_points]
      - validation: series[-cv_points:]

    Each model forecasts cv_points steps using training data,
    and also forecasts horizon steps using the full series.
    """
    if cv_points is None:
        cv_points = min(horizon, max(10, len(series) // 5))

    if len(series) < cv_points + 20:
        logger.warning("Series too short for cross-validation, using full series")
        cv_points = min(horizon, len(series) // 4)

    training = series.iloc[:-cv_points]
    validation = series.iloc[-cv_points:]
    val_actuals = validation.values.astype(float)

    if model_names is None:
        available = get_available_forecasters()
        model_names = [m for m in available if m not in ("llm", "baseline")]

    MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}
    results = {}

    for model_name in model_names:
        try:
            kwargs = {}
            if model_name in MODELS_WITH_SIZE:
                kwargs["model_size"] = model_size

            fc = create_forecaster(model_name, **kwargs)
            if fc is None:
                continue

            # Cross-validation: predict on held-out set
            cv_result = fc.forecast(
                series=training,
                horizon=cv_points,
                context_length=min(512, len(training)),
            )
            cv_preds = cv_result.forecast if cv_result.success else None

            # Forward forecast: predict using full series
            fwd_result = fc.forecast(
                series=series,
                horizon=horizon,
                context_length=min(512, len(series)),
            )
            fwd_preds = fwd_result.forecast if fwd_result.success else None

            if cv_preds is not None and fwd_preds is not None:
                # Compute MAE on validation set
                cv_array = np.array(cv_preds[:len(val_actuals)], dtype=float)
                mae = float(np.mean(np.abs(cv_array - val_actuals[:len(cv_array)])))

                # Extract quantiles from metadata if available
                quantiles = None
                if fwd_result.metadata and "quantiles" in fwd_result.metadata:
                    quantiles = fwd_result.metadata["quantiles"]

                results[model_name] = CrossValResult(
                    model_name=model_name,
                    mae=mae,
                    cv_predictions=cv_preds[:len(val_actuals)],
                    forecast=fwd_preds,
                    quantiles=quantiles,
                )
            else:
                results[model_name] = CrossValResult(
                    model_name=model_name,
                    mae=float("inf"),
                    cv_predictions=[],
                    forecast=[],
                    success=False,
                    error=f"Forecast failed: cv={cv_result.error}, fwd={fwd_result.error}",
                )

        except Exception as e:
            logger.warning(f"CV for {model_name} failed: {e}")
            results[model_name] = CrossValResult(
                model_name=model_name,
                mae=float("inf"),
                cv_predictions=[],
                forecast=[],
                success=False,
                error=str(e),
            )

    return results


def build_llm_selection_prompt(
    series: pd.Series,
    cv_results: Dict[str, CrossValResult],
    mae_ranking: List[Tuple[str, float]],
    horizon: int,
) -> str:
    """
    Build prompt for LLM model selection.

    Shows the LLM:
    - Normalized + downsampled history
    - Each model's forward predictions
    - Each model's CV predictions vs actuals
    - MAE ranking from cross-validation
    """
    values = series.values.astype(float)
    norm_values, mean, std = normalize_series(values)
    downsampled = downsample_for_llm(norm_values, max_points=40)

    history_str = ",".join([f"{v:.3f}" for v in downsampled])

    # Build candidate predictions section
    pred_lines = []
    for model_name, result in cv_results.items():
        if result.success and result.forecast:
            norm_fwd = [(v - mean) / std for v in result.forecast]
            ds_fwd = downsample_for_llm(np.array(norm_fwd), max_points=40)
            pred_str = ",".join([f"{v:.3f}" for v in ds_fwd])
            pred_lines.append(f'  "{model_name}": "{pred_str}"')

    # Build CV section
    cv_lines = []
    cv_actuals = series.iloc[-min(horizon, len(series) // 5):].values.astype(float)
    norm_actuals = [(v - mean) / std for v in cv_actuals]
    ds_actuals = downsample_for_llm(np.array(norm_actuals), max_points=40)
    actuals_str = ",".join([f"{v:.3f}" for v in ds_actuals])

    for model_name, result in cv_results.items():
        if result.success and result.cv_predictions:
            norm_cv = [(v - mean) / std for v in result.cv_predictions]
            ds_cv = downsample_for_llm(np.array(norm_cv), max_points=40)
            cv_str = ",".join([f"{v:.3f}" for v in ds_cv])
            cv_lines.append(f'  "{model_name}": "{cv_str}"')

    # MAE ranking string
    ranking_str = " < ".join([f"{m} (MAE={e:.4f})" for m, e in mae_ranking if e < float("inf")])

    model_names = [m for m, _ in mae_ranking if m in cv_results and cv_results[m].success]

    prompt = f"""You are given a normalized time series and predictions from candidate models.

HISTORY (normalized):
{history_str}

CANDIDATE FUTURE PREDICTIONS:
{chr(10).join(pred_lines)}

CROSS-VALIDATION GROUND TRUTH (last portion of history):
{actuals_str}

CROSS-VALIDATION PREDICTIONS:
{chr(10).join(cv_lines)}

CROSS-VALIDATION ERROR RANKING (best to worst):
{ranking_str}

Analyze the future predictions and their cross-validation performance.
Select the optimal model for this forecast. Consider:
- Cross-validation MAE (lower is better)
- Whether predictions look reasonable given the history
- Pattern consistency between CV and forward predictions

You may select "mixture" for an equal-weight blend of all models.

Enclose the name of the best model with \\boxed{{ and }}.
Available: {', '.join(model_names)}, mixture"""

    return prompt


def parse_llm_selection(response: str, available_models: List[str]) -> str:
    """Parse LLM response for \\boxed{model_name} pattern."""
    match = re.search(r'\\boxed\{(\w[\w-]*)\}', response)
    if match:
        selected = match.group(1).lower().strip()
        if selected in available_models or selected == "mixture":
            return selected

    # Fallback: look for model names in the response
    for model in available_models:
        if model.lower() in response.lower():
            return model

    return "mixture"


def select_model_with_cv(
    series: pd.Series,
    horizon: int,
    provider=None,
    model_names: Optional[List[str]] = None,
    model_size: str = "small",
    cv_points: Optional[int] = None,
    diversity_threshold: float = 0.01,
) -> CVSelectionResult:
    """
    Full cross-validation-based model selection pipeline.

    1. Run all candidate models for CV and forward forecast
    2. Compute MAE ranking
    3. Check prediction diversity
    4. If diverse and LLM available: ask LLM to select
    5. If not diverse: auto-select "mixture"
    6. Return CVSelectionResult
    """
    logger.info("Running cross-validation model selection")

    # Step 1: Run cross-validation
    cv_results = run_cross_validation(
        series=series,
        horizon=horizon,
        cv_points=cv_points,
        model_names=model_names,
        model_size=model_size,
    )

    successful = {k: v for k, v in cv_results.items() if v.success}
    if not successful:
        logger.warning("No models succeeded in CV, falling back to mixture")
        return CVSelectionResult(
            selected_model="mixture",
            selection_method="fallback",
            cv_results=cv_results,
        )

    # Step 2: Compute MAE ranking
    mae_ranking = sorted(
        [(name, r.mae) for name, r in successful.items()],
        key=lambda x: x[1]
    )

    # Step 3: Check prediction diversity
    forecasts = {k: v.forecast for k, v in successful.items() if v.forecast}
    is_diverse, diversity_score = check_prediction_diversity(
        forecasts, threshold=diversity_threshold
    )

    # Step 4: Select model
    if not is_diverse:
        logger.info(f"Predictions not diverse (score={diversity_score:.4f}), using mixture")
        return CVSelectionResult(
            selected_model="mixture",
            selection_method="mixture",
            cv_results=cv_results,
            mae_ranking=mae_ranking,
            diversity_score=diversity_score,
            is_diverse=False,
            llm_reasoning="Models show high agreement. Equal-weight mixture used.",
        )

    if provider is not None:
        # LLM-based selection
        try:
            prompt = build_llm_selection_prompt(series, successful, mae_ranking, horizon)
            response = provider.generate(prompt)
            available = list(successful.keys())
            selected = parse_llm_selection(response, available)

            logger.info(f"LLM selected model: {selected}")
            return CVSelectionResult(
                selected_model=selected,
                selection_method="llm",
                cv_results=cv_results,
                mae_ranking=mae_ranking,
                diversity_score=diversity_score,
                is_diverse=True,
                llm_reasoning=response[:500],
            )
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, using best MAE")

    # Fallback: use model with lowest MAE
    best_model = mae_ranking[0][0]
    logger.info(f"Using best MAE model: {best_model}")
    return CVSelectionResult(
        selected_model=best_model,
        selection_method="best_mae",
        cv_results=cv_results,
        mae_ranking=mae_ranking,
        diversity_score=diversity_score,
        is_diverse=is_diverse,
    )
