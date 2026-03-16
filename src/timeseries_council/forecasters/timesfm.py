# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Google TimesFM forecaster.
"""

from typing import Optional, Callable
import pandas as pd

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger
from ..exceptions import ForecasterError
from ..utils import get_device

logger = get_logger(__name__)


class TimesFMForecaster(BaseForecaster):
    """Google TimesFM foundation model forecaster."""

    def __init__(
        self,
        model_size: str = "200m",
        device: str = None
    ):
        """
        Initialize TimesFM forecaster.

        Args:
            model_size: Model size ('200m')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_size = model_size
        self.device = get_device(device)
        self._model = None

        logger.info(f"Initialized TimesFMForecaster: {model_size} on {self.device}")

    @property
    def name(self) -> str:
        return f"TimesFM-{self.model_size}"

    @property
    def description(self) -> str:
        return f"Google TimesFM foundation model ({self.model_size})"

    def _get_model(self):
        """Lazy load the TimesFM model."""
        if self._model is None:
            try:
                import timesfm

                logger.info(f"Loading TimesFM model: {self.model_size}")

                # New API uses hparams and checkpoint objects
                # Set horizon_len large enough for most use cases (512 max)
                hparams = timesfm.TimesFmHparams(
                    context_len=512,
                    horizon_len=512,  # Set high, we'll slice output to actual horizon
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend="cpu" if self.device == "cpu" else "gpu",
                )
                checkpoint = timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
                )
                self._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            except ImportError:
                raise ForecasterError(
                    "timesfm not installed. Run: pip install timesfm",
                    {"forecaster": "timesfm"}
                )
        return self._model

    def _statistical_fallback(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None
    ) -> ForecastResult:
        """Statistical fallback when TimesFM is not available."""
        import numpy as np

        logger.warning("Using statistical fallback for TimesFM")

        ctx_len = context_length or min(168, len(series))
        context = series.tail(ctx_len)

        # Simple trend + seasonality estimation
        trend = np.polyfit(range(len(context)), context.values, 1)[0]
        mean_val = context.mean()
        std_val = context.std()

        # Generate forecast with trend
        forecast_values = []
        for i in range(horizon):
            predicted = mean_val + trend * (len(context) + i)
            forecast_values.append(round(float(predicted), 2))

        # Generate timestamps - use robust frequency inference
        from ..utils import infer_frequency
        last_timestamp = series.index[-1]
        freq = infer_frequency(series)
        # Generate future dates starting after last timestamp using periods + 1 and skip first
        future_index = pd.date_range(
            start=last_timestamp,
            periods=horizon + 1,
            freq=freq
        )[1:]  # Skip the first one which is last_timestamp
        timestamps = [str(t) for t in future_index]

        return ForecastResult(
            success=True,
            forecast=forecast_values,
            timestamps=timestamps,
            uncertainty=[round(float(std_val), 2)] * horizon,
            horizon=horizon,
            model_name="timesfm-fallback",
            metadata={"context_length": ctx_len, "fallback": True}
        )

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using TimesFM model."""
        import numpy as np

        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading TimesFM model...", 0.1)

        try:
            model = self._get_model()
        except (ImportError, ForecasterError) as e:
            logger.warning(f"TimesFM not available: {e}")
            return self._statistical_fallback(series, horizon, context_length)

        try:
            self._report_progress(progress_callback, "Preparing data...", 0.3)

            # Prepare context
            ctx_len = context_length or min(512, len(series))
            context = series.tail(ctx_len).values

            self._report_progress(progress_callback, "Generating forecast...", 0.5)

            # TimesFM expects 2D array: (batch, time)
            context_array = np.array([context])

            # Infer frequency
            freq = pd.infer_freq(series.index)
            freq_map = {
                'D': 0,  # Daily
                'W': 1,  # Weekly
                'M': 2,  # Monthly
                'Q': 3,  # Quarterly
                'Y': 4,  # Yearly
                'H': 5,  # Hourly
            }
            # Extract the unit from frequency (e.g., 'H' from '2H')
            freq_char = next((c for c in freq if c.isalpha()), 'D') if freq else 'D'
            freq_id = freq_map.get(freq_char, 0)

            # Generate forecast (horizon is set by hparams.horizon_len, we slice output)
            point_forecast, quantile_forecast = model.forecast(
                context_array,
                freq=[freq_id],
            )

            self._report_progress(progress_callback, "Processing results...", 0.8)

            # Extract results
            forecast_values = point_forecast[0, :horizon]

            # Uncertainty from quantiles
            uncertainty = None
            if quantile_forecast is not None:
                # Use spread between 10th and 90th percentile
                q10 = quantile_forecast[0, :horizon, 0]  # 10th percentile
                q90 = quantile_forecast[0, :horizon, -1]  # 90th percentile
                uncertainty = ((q90 - q10) / 2).tolist()

            # Generate timestamps - use robust frequency inference
            from ..utils import infer_frequency
            last_timestamp = series.index[-1]
            freq_str = infer_frequency(series)
            # Generate future dates starting after last timestamp using periods + 1 and skip first
            future_index = pd.date_range(
                start=last_timestamp,
                periods=horizon + 1,
                freq=freq_str
            )[1:]  # Skip the first one which is last_timestamp
            timestamps = [str(t) for t in future_index]

            self._report_progress(progress_callback, "Forecast complete", 1.0)

            logger.info(f"TimesFM forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=np.round(forecast_values, 2).tolist(),
                timestamps=timestamps,
                uncertainty=uncertainty,
                horizon=horizon,
                model_name=f"timesfm-{self.model_size}",
                metadata={
                    "context_length": ctx_len,
                    "frequency": freq_str
                }
            )

        except Exception as e:
            logger.error(f"TimesFM forecast failed: {e}")
            return ForecastResult(success=False, error=str(e))
