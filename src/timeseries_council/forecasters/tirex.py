# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
NX-AI TiRex foundation model forecaster.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger
from ..utils import get_device

logger = get_logger(__name__)

# Model ID mapping - TiRex currently only has one model (35M params)
TIREX_MODELS = {
    "small": "NX-AI/TiRex",
    "base": "NX-AI/TiRex",
    "large": "NX-AI/TiRex",
}


class TiRexForecaster(BaseForecaster):
    """NX-AI TiRex foundation model forecaster with quantile predictions."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = None
    ):
        """
        Initialize TiRex forecaster.

        Args:
            model_size: Model size ('small', 'base', 'large')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_size = model_size.lower()
        self.device = get_device(device)
        self._model = None
        logger.info(f"Initialized TiRexForecaster: {self.model_size} on {self.device}")

    @property
    def name(self) -> str:
        return "TiRex"

    @property
    def description(self) -> str:
        return "NX-AI TiRex foundation model (35M params)"

    def _get_model_id(self) -> str:
        """Get model ID."""
        return TIREX_MODELS.get(self.model_size, TIREX_MODELS["base"])

    def _load_model(self):
        """Lazy-load TiRex model."""
        if self._model is None:
            from tirex import ForecastModel, load_model
            model_id = self._get_model_id()
            logger.info(f"Loading TiRex from: {model_id}")
            self._model = load_model(model_id, device=self.device)
        return self._model

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using TiRex model."""
        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading TiRex model...", 0.1)

        try:
            import torch
        except ImportError as e:
            logger.error(f"PyTorch not installed: {e}")
            return ForecastResult(
                success=False,
                error="PyTorch not installed. Run: pip install torch"
            )

        try:
            # Prepare context
            ctx_len = context_length or min(4000, len(series))
            context = series.tail(ctx_len).values.astype(np.float32)

            self._report_progress(progress_callback, "Preparing forecast...", 0.3)

            try:
                model = self._load_model()
                USE_TIREX = True
            except (ImportError, Exception) as e:
                logger.warning(f"TiRex not available ({e}), using statistical fallback")
                USE_TIREX = False

            if USE_TIREX:
                self._report_progress(progress_callback, "Generating forecast...", 0.5)

                quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                with torch.no_grad():
                    context_tensor = torch.tensor(context).unsqueeze(0)  # [1, context_len]
                    quantile_forecast, mean_forecast = model.forecast(
                        context=context_tensor,
                        prediction_length=horizon,
                    )

                # Extract predictions: quantile_forecast shape [1, horizon, 9], mean shape [1, horizon]
                quantiles = quantile_forecast[0, :horizon, :].cpu().numpy()
                mean_values = mean_forecast[0, :horizon].cpu().numpy()

                # Use mean predictions as primary forecast
                forecast_values = mean_values.tolist()

                # Uncertainty from 10th and 90th percentiles
                q10 = quantiles[:, 0]
                q90 = quantiles[:, 8]
                uncertainty = ((q90 - q10) / 2).tolist()

                # Store full quantiles in metadata
                quantile_data = {}
                for q_idx, q_level in enumerate(quantile_levels):
                    quantile_data[str(q_level)] = quantiles[:, q_idx].tolist()

            else:
                # Statistical fallback
                logger.warning("Using statistical fallback for TiRex")
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(context, size=min(7, len(context)))

                if len(context) >= 7:
                    trend = (smoothed[-1] - smoothed[-7]) / 7
                else:
                    trend = 0

                last_val = context[-1]
                forecast_values = [round(last_val + trend * (i + 1), 2) for i in range(horizon)]
                hist_std = float(np.std(context[-14:]) if len(context) >= 14 else np.std(context))
                uncertainty = [round(hist_std, 2)] * horizon
                quantile_data = {}

            self._report_progress(progress_callback, "Processing results...", 0.8)

            # Generate timestamps
            from ..utils import infer_frequency
            last_timestamp = series.index[-1]
            freq = infer_frequency(series)
            future_index = pd.date_range(
                start=last_timestamp,
                periods=horizon + 1,
                freq=freq
            )[1:]
            timestamps = [str(t) for t in future_index]

            self._report_progress(progress_callback, "Forecast complete", 1.0)

            logger.info(f"TiRex forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=[round(v, 2) for v in forecast_values],
                timestamps=timestamps,
                uncertainty=[round(v, 2) for v in uncertainty],
                horizon=horizon,
                model_name="tirex",
                metadata={
                    "context_length": ctx_len,
                    "device": self.device,
                    "fallback_used": not USE_TIREX,
                    "quantiles": quantile_data,
                }
            )

        except Exception as e:
            logger.error(f"TiRex forecast failed: {e}")
            import traceback
            return ForecastResult(
                success=False,
                error=str(e),
                metadata={"traceback": traceback.format_exc()}
            )
