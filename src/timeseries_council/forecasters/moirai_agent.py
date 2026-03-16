# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
MoiraiAgent-style forecaster: LLM-driven expert selection across multiple candidate models.

Based on the MoiraiAgent framework (Salesforce Research), this forecaster:
1. Runs multiple candidate forecasters in parallel
2. Cross-validates on held-out data to compute per-model MAE
3. Checks prediction diversity (returns mixture if models agree)
4. Uses an LLM to select the best model when predictions diverge
5. Applies offset adjustment to align the mixture with the selected model

Reference: https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-agent
"""

from typing import Optional, Callable, List, Dict, Tuple
import pandas as pd
import numpy as np
import re

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger

logger = get_logger(__name__)

# Default candidate models (same pool as MoiraiAgent paper)
DEFAULT_CANDIDATES = ["moirai", "chronos", "tirex", "timesfm"]

# LLM prompt for expert selection (follows MoiraiAgent prompt structure)
SELECTION_PROMPT = """You are a time series forecasting expert. Given a historical time series, cross-validation errors, and predictions from multiple forecasting models, select the single best model.

HISTORICAL SERIES (last {n_points} points, normalized):
{history}

CROSS-VALIDATION RESULTS (MAE on held-out validation set):
{cv_results}

MODEL PREDICTIONS (next {horizon} steps):
{predictions}

RANKING BY CV ERROR (best to worst):
{ranking}

Based on the series characteristics, cross-validation performance, and predicted trajectories, which model should be used?

Rules:
- If models largely agree, select "mixture" for a blended forecast
- If one model clearly fits the data pattern better, select that model
- Consider both CV error and prediction trajectory shape

Respond with your selection inside \\boxed{{model_name}}. For example: \\boxed{{chronos}}"""


class MoiraiAgentForecaster(BaseForecaster):
    """
    MoiraiAgent-style expert selection forecaster.

    Runs multiple candidate forecasters, evaluates them via cross-validation,
    and uses an LLM to select the best model for the given series. When no
    LLM provider is available, falls back to selecting the model with the
    lowest cross-validation error.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = None,
        candidate_models: Optional[List[str]] = None,
        provider=None,
        diversity_threshold: float = 0.01,
    ):
        """
        Initialize MoiraiAgent forecaster.

        Args:
            model_size: Size for candidate foundation models ('small', 'base', 'large')
            device: Device for model inference ('cpu', 'cuda', or None for auto)
            candidate_models: List of forecaster names to use as candidates.
                Defaults to moirai, chronos, tirex, timesfm.
            provider: LLM provider for expert selection (optional, falls back to
                lowest-CV-error selection if not provided)
            diversity_threshold: Normalized MAE threshold below which predictions
                are considered similar enough to skip LLM selection (default 0.01)
        """
        from ..utils import get_device

        self.model_size = model_size.lower()
        self.device = get_device(device)
        self.candidate_models = candidate_models or DEFAULT_CANDIDATES.copy()
        self.provider = provider
        self.diversity_threshold = diversity_threshold
        logger.info(
            f"Initialized MoiraiAgentForecaster: candidates={self.candidate_models}, "
            f"model_size={self.model_size}"
        )

    @property
    def name(self) -> str:
        return "MoiraiAgent"

    @property
    def description(self) -> str:
        return (
            f"MoiraiAgent expert selection across "
            f"{len(self.candidate_models)} candidate models"
        )

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ForecastResult:
        """
        Generate forecast using MoiraiAgent expert selection pipeline.

        Pipeline:
        1. Split series into train/validation for cross-validation
        2. Run candidate forecasters on training set
        3. Compute MAE on validation set per model
        4. Run all candidates on full series for final predictions
        5. Check prediction diversity
        6. If diverse and LLM available: LLM selects best model
        7. Apply offset adjustment and return
        """
        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Starting MoiraiAgent pipeline...", 0.05)

        try:
            from . import create_forecaster

            # Determine available candidates
            from . import get_available_forecasters
            available = get_available_forecasters()
            candidates = [m for m in self.candidate_models if m in available]

            if not candidates:
                logger.warning("No candidate forecasters available, using baseline")
                candidates = ["zscore_baseline"]

            # Models that accept model_size
            MODELS_WITH_SIZE = {"moirai", "chronos", "timesfm", "tirex"}

            # --- Step 1: Cross-validation ---
            self._report_progress(progress_callback, "Cross-validating candidates...", 0.1)

            cv_horizon = min(horizon, max(5, len(series) // 10))
            train_series = series.iloc[:-cv_horizon]
            val_actual = series.iloc[-cv_horizon:].values

            cv_results: Dict[str, float] = {}  # model -> MAE
            cv_predictions: Dict[str, List[float]] = {}

            for model_name in candidates:
                try:
                    if model_name in MODELS_WITH_SIZE:
                        fc = create_forecaster(model_name, model_size=self.model_size)
                    else:
                        fc = create_forecaster(model_name)

                    result = fc.forecast(
                        series=train_series,
                        horizon=cv_horizon,
                        context_length=context_length,
                    )

                    if result.success and result.forecast:
                        pred = np.array(result.forecast[:cv_horizon])
                        actual = val_actual[:len(pred)]
                        mae = float(np.mean(np.abs(pred - actual)))
                        cv_results[model_name] = mae
                        cv_predictions[model_name] = result.forecast[:cv_horizon]
                except Exception as e:
                    logger.warning(f"CV failed for {model_name}: {e}")

            if not cv_results:
                logger.warning("All candidates failed CV, using statistical fallback")
                return self._statistical_fallback(series, horizon, progress_callback)

            self._report_progress(progress_callback, "Running final predictions...", 0.4)

            # --- Step 2: Run all candidates on full series ---
            final_predictions: Dict[str, List[float]] = {}
            final_timestamps = None
            final_uncertainties: Dict[str, List[float]] = {}

            for model_name in cv_results.keys():
                try:
                    if model_name in MODELS_WITH_SIZE:
                        fc = create_forecaster(model_name, model_size=self.model_size)
                    else:
                        fc = create_forecaster(model_name)

                    result = fc.forecast(
                        series=series,
                        horizon=horizon,
                        context_length=context_length,
                    )

                    if result.success and result.forecast:
                        final_predictions[model_name] = result.forecast[:horizon]
                        if result.timestamps and final_timestamps is None:
                            final_timestamps = result.timestamps
                        if result.uncertainty:
                            final_uncertainties[model_name] = result.uncertainty[:horizon]
                except Exception as e:
                    logger.warning(f"Final prediction failed for {model_name}: {e}")

            if not final_predictions:
                return self._statistical_fallback(series, horizon, progress_callback)

            self._report_progress(progress_callback, "Evaluating predictions...", 0.6)

            # --- Step 3: Compute mixture ---
            pred_arrays = [
                np.array(p) for p in final_predictions.values()
            ]
            min_len = min(len(p) for p in pred_arrays)
            pred_arrays = [p[:min_len] for p in pred_arrays]
            mixture = np.mean(pred_arrays, axis=0)

            # --- Step 4: Diversity check ---
            mae_ranking = sorted(cv_results.items(), key=lambda x: x[1])
            diversity_score = self._compute_diversity(pred_arrays, series)

            selected_model = "mixture"
            selection_method = "diversity_low"
            llm_reasoning = ""

            if diversity_score > self.diversity_threshold:
                # Predictions are diverse, need to select
                self._report_progress(
                    progress_callback, "Selecting best model...", 0.7
                )

                if self.provider is not None:
                    # LLM-driven selection
                    selected, reasoning = self._llm_select(
                        series, horizon, cv_results, final_predictions, mae_ranking
                    )
                    if selected and selected in final_predictions:
                        selected_model = selected
                        selection_method = "llm"
                        llm_reasoning = reasoning
                    elif selected == "mixture":
                        selected_model = "mixture"
                        selection_method = "llm_mixture"
                        llm_reasoning = reasoning
                    else:
                        # LLM failed or returned unknown model, use best CV
                        selected_model = mae_ranking[0][0]
                        selection_method = "cv_best"
                else:
                    # No LLM, use best CV model
                    selected_model = mae_ranking[0][0]
                    selection_method = "cv_best"

            self._report_progress(progress_callback, "Applying offset adjustment...", 0.85)

            # --- Step 5: Offset adjustment ---
            if selected_model != "mixture" and selected_model in final_predictions:
                forecast_values = self._apply_offset_adjustment(
                    mixture, np.array(final_predictions[selected_model][:min_len])
                ).tolist()
            else:
                forecast_values = mixture.tolist()

            # Compute uncertainty from model spread
            uncertainty = np.std(pred_arrays, axis=0).tolist()

            # Generate timestamps if missing
            if final_timestamps is None:
                from ..utils import infer_frequency
                freq = infer_frequency(series)
                future_index = pd.date_range(
                    start=series.index[-1],
                    periods=min_len + 1,
                    freq=freq,
                )[1:]
                final_timestamps = [str(t) for t in future_index]

            self._report_progress(progress_callback, "MoiraiAgent complete", 1.0)

            logger.info(
                f"MoiraiAgent selected '{selected_model}' via {selection_method} "
                f"(diversity={diversity_score:.4f})"
            )

            return ForecastResult(
                success=True,
                forecast=[round(v, 2) for v in forecast_values],
                timestamps=final_timestamps[:min_len],
                uncertainty=[round(v, 2) for v in uncertainty],
                horizon=min_len,
                model_name=f"moirai-agent ({selected_model})",
                metadata={
                    "selected_model": selected_model,
                    "selection_method": selection_method,
                    "diversity_score": round(diversity_score, 4),
                    "cv_mae_ranking": [
                        {"model": m, "mae": round(e, 4)} for m, e in mae_ranking
                    ],
                    "candidates_used": list(final_predictions.keys()),
                    "llm_reasoning": llm_reasoning,
                    "individual_forecasts": {
                        m: [round(v, 2) for v in p]
                        for m, p in final_predictions.items()
                    },
                },
            )

        except Exception as e:
            logger.error(f"MoiraiAgent forecast failed: {e}")
            import traceback
            return ForecastResult(
                success=False,
                error=str(e),
                metadata={"traceback": traceback.format_exc()},
            )

    def _compute_diversity(
        self, pred_arrays: List[np.ndarray], series: pd.Series
    ) -> float:
        """
        Compute normalized prediction diversity score.

        Returns the mean pairwise MAE between predictions, normalized by the
        series standard deviation. A low score means models agree.
        """
        if len(pred_arrays) < 2:
            return 0.0

        series_std = float(np.std(series.values))
        if series_std == 0:
            return 0.0

        pairwise_maes = []
        for i in range(len(pred_arrays)):
            for j in range(i + 1, len(pred_arrays)):
                mae = float(np.mean(np.abs(pred_arrays[i] - pred_arrays[j])))
                pairwise_maes.append(mae)

        return float(np.mean(pairwise_maes)) / series_std

    def _apply_offset_adjustment(
        self, mixture: np.ndarray, selected: np.ndarray
    ) -> np.ndarray:
        """
        Shift the mixture forecast to align with the selected model's median.

        This preserves the ensemble's smoothness while centering predictions
        around the selected model's trajectory.
        """
        offset = np.median(selected) - np.median(mixture)
        return mixture + offset

    def _llm_select(
        self,
        series: pd.Series,
        horizon: int,
        cv_results: Dict[str, float],
        predictions: Dict[str, List[float]],
        mae_ranking: List[Tuple[str, float]],
    ) -> Tuple[Optional[str], str]:
        """
        Use the LLM provider to select the best forecasting model.

        Returns (selected_model_name, reasoning_text).
        """
        try:
            # Downsample history for prompt
            values = series.values
            max_points = 40
            if len(values) > max_points:
                indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
                downsampled = values[indices]
            else:
                downsampled = values

            # Normalize for display
            mean_val = np.mean(downsampled)
            std_val = np.std(downsampled)
            if std_val > 0:
                normalized = (downsampled - mean_val) / std_val
            else:
                normalized = downsampled - mean_val

            history_str = ", ".join(f"{v:.2f}" for v in normalized)

            # CV results string
            cv_str = "\n".join(
                f"  {m}: MAE = {e:.4f}" for m, e in mae_ranking
            )

            # Predictions string (also normalized)
            pred_str_parts = []
            for m, p in predictions.items():
                if std_val > 0:
                    norm_p = [(v - mean_val) / std_val for v in p[:horizon]]
                else:
                    norm_p = [v - mean_val for v in p[:horizon]]
                pred_str_parts.append(
                    f"  {m}: [{', '.join(f'{v:.2f}' for v in norm_p)}]"
                )
            pred_str = "\n".join(pred_str_parts)

            # Ranking string
            rank_str = " > ".join(f"{m} ({e:.4f})" for m, e in mae_ranking)

            prompt = SELECTION_PROMPT.format(
                n_points=len(normalized),
                history=history_str,
                cv_results=cv_str,
                predictions=pred_str,
                horizon=horizon,
                ranking=rank_str,
            )

            response = self.provider.generate(prompt)

            # Parse \boxed{model_name}
            match = re.search(r"\\boxed\{(\w[\w\-]*)\}", response)
            if match:
                selected = match.group(1).lower().strip()
                return selected, response
            else:
                # Try to find model name in the response
                available = list(predictions.keys()) + ["mixture"]
                for model in available:
                    if model.lower() in response.lower():
                        return model, response

                logger.warning(f"Could not parse LLM selection from response")
                return None, response

        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")
            return None, str(e)

    def _statistical_fallback(
        self,
        series: pd.Series,
        horizon: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ForecastResult:
        """Fallback when no candidate models are available."""
        logger.warning("Using statistical fallback for MoiraiAgent")
        self._report_progress(progress_callback, "Using statistical fallback...", 0.5)

        context = series.values.astype(np.float32)

        from scipy.ndimage import uniform_filter1d

        smoothed = uniform_filter1d(context, size=min(7, len(context)))

        if len(context) >= 7:
            trend = (smoothed[-1] - smoothed[-7]) / 7
        else:
            trend = 0

        last_val = context[-1]
        forecast_values = [
            round(last_val + trend * (i + 1), 2) for i in range(horizon)
        ]
        hist_std = float(
            np.std(context[-14:]) if len(context) >= 14 else np.std(context)
        )
        uncertainty = [round(hist_std, 2)] * horizon

        from ..utils import infer_frequency

        freq = infer_frequency(series)
        future_index = pd.date_range(
            start=series.index[-1], periods=horizon + 1, freq=freq
        )[1:]
        timestamps = [str(t) for t in future_index]

        self._report_progress(progress_callback, "Fallback complete", 1.0)

        return ForecastResult(
            success=True,
            forecast=forecast_values,
            timestamps=timestamps,
            uncertainty=uncertainty,
            horizon=horizon,
            model_name="moirai-agent (fallback)",
            metadata={"fallback_used": True, "selection_method": "fallback"},
        )
