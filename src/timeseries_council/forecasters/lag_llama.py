# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Lag-Llama forecaster.
"""

from typing import Optional, Callable
import pandas as pd

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger
from ..exceptions import ForecasterError
from ..utils import get_device

logger = get_logger(__name__)


class LagLlamaForecaster(BaseForecaster):
    """Lag-Llama foundation model forecaster."""

    def __init__(
        self,
        device: str = None,
        context_length: int = 32,
        use_rope_scaling: bool = False
    ):
        """
        Initialize Lag-Llama forecaster.

        Args:
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            context_length: Context length for the model
            use_rope_scaling: Whether to use RoPE scaling for longer contexts
        """
        self.device = get_device(device)
        self.default_context_length = context_length
        self.use_rope_scaling = use_rope_scaling
        self._model = None
        self._config = None

        logger.info(f"Initialized LagLlamaForecaster on {self.device}")

    @property
    def name(self) -> str:
        return "Lag-Llama"

    @property
    def description(self) -> str:
        return "Lag-Llama foundation model for probabilistic time series forecasting"

    def _get_model(self):
        """Lazy load the Lag-Llama model."""
        if self._model is None:
            try:
                from lag_llama.gluon.estimator import LagLlamaEstimator
                import torch

                logger.info("Loading Lag-Llama model...")

                # Download from HuggingFace if needed
                from huggingface_hub import hf_hub_download

                ckpt_path = hf_hub_download(
                    repo_id="time-series-foundation-models/Lag-Llama",
                    filename="lag-llama.ckpt"
                )

                # Load checkpoint to get hyper_parameters (PyTorch 2.6+ needs weights_only=False)
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    hyper_params = ckpt.get("hyper_parameters", {})
                    logger.info(f"Checkpoint hyper_parameters: {hyper_params}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint for inspection: {e}")
                    hyper_params = {}

                # Load estimator with architecture matching the checkpoint
                device = torch.device(self.device)

                # Build kwargs matching checkpoint architecture
                # Use hyper_parameters from checkpoint if available, else defaults
                estimator_kwargs = {
                    "prediction_length": 1,  # Will be overridden per forecast call
                    "context_length": hyper_params.get("context_length", self.default_context_length),
                    "device": device,
                    "num_parallel_samples": 100,
                    "ckpt_path": ckpt_path,
                    # Architecture params from checkpoint (must match for weight loading)
                    "n_layer": hyper_params.get("n_layer", 8),
                    "n_embd_per_head": hyper_params.get("n_embd_per_head", 16),
                    "n_head": hyper_params.get("n_head", 9),
                    "time_feat": hyper_params.get("time_feat", True),
                    "scaling": hyper_params.get("scaling", "robust"),
                }
                if self.use_rope_scaling:
                    estimator_kwargs["rope_scaling"] = {"type": "linear", "factor": 2.0}

                estimator = LagLlamaEstimator(**estimator_kwargs)

                self._model = estimator

            except ImportError as e:
                raise ForecasterError(
                    f"Lag-Llama dependencies not installed: {e}. "
                    "Run: pip install lag-llama gluonts",
                    {"forecaster": "lag-llama"}
                )
        return self._model

    def _statistical_fallback(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None
    ) -> ForecastResult:
        """Statistical fallback when Lag-Llama is not available."""
        import numpy as np

        logger.warning("Using statistical fallback for Lag-Llama")

        ctx_len = context_length or self.default_context_length
        context = series.tail(ctx_len)

        # Simple trend + noise estimation for probabilistic-like output
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
            model_name="lag-llama-fallback",
            metadata={"context_length": ctx_len, "fallback": True}
        )

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using Lag-Llama model."""
        import numpy as np

        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading Lag-Llama model...", 0.1)

        try:
            estimator = self._get_model()
        except (ImportError, ForecasterError) as e:
            logger.warning(f"Lag-Llama not available: {e}")
            return self._statistical_fallback(series, horizon, context_length)

        try:
            from gluonts.dataset.pandas import PandasDataset

            self._report_progress(progress_callback, "Preparing data...", 0.3)

            # Create GluonTS dataset with timestamp column
            ctx_len = context_length or self.default_context_length
            context_series = series.tail(ctx_len + horizon).copy()

            # Log input data characteristics
            logger.info(f"Input series: len={len(series)}, dtype={series.dtype}, index_type={type(series.index).__name__}")

            # Ensure we have a DatetimeIndex
            if not isinstance(context_series.index, pd.DatetimeIndex):
                logger.warning("Converting series index to DatetimeIndex")
                context_series.index = pd.to_datetime(context_series.index)

            # Ensure data is clean
            context_series = context_series.dropna()
            if len(context_series) < 10:
                raise ForecasterError(
                    f"Insufficient data after cleaning: {len(context_series)} points",
                    {"forecaster": "lag-llama"}
                )

            # Convert to float32 for PyTorch compatibility
            target_values = context_series.values.astype(np.float32)

            logger.info(f"Context series: len={len(context_series)}, start={context_series.index[0]}")

            # Use ListDataset which is more reliable than PandasDataset
            from gluonts.dataset.common import ListDataset
            freq = pd.infer_freq(context_series.index) or "D"
            dataset = ListDataset(
                [{"start": context_series.index[0], "target": target_values}],
                freq=freq
            )

            self._report_progress(progress_callback, "Creating predictor...", 0.4)

            # Update prediction length and create predictor
            estimator.prediction_length = horizon

            # Use the trained predictor directly from the estimator
            # The ckpt_path in estimator will be used to create the trained model
            predictor = estimator.create_predictor(estimator.create_transformation(), estimator.create_lightning_module())

            if predictor is None:
                raise ForecasterError("Failed to create Lag-Llama predictor", {"forecaster": "lag-llama"})

            self._report_progress(progress_callback, "Generating forecast...", 0.6)

            # Generate forecasts
            forecasts = list(predictor.predict(dataset))

            if not forecasts:
                return ForecastResult(
                    success=False,
                    error="No forecasts generated"
                )

            self._report_progress(progress_callback, "Processing results...", 0.8)

            forecast = forecasts[0]

            # Get median and uncertainty
            median_forecast = forecast.median
            uncertainty = forecast.quantile(0.9) - forecast.quantile(0.1)
            uncertainty = (uncertainty / 2).tolist()

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

            self._report_progress(progress_callback, "Forecast complete", 1.0)

            logger.info(f"Lag-Llama forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=np.round(median_forecast, 2).tolist(),
                timestamps=timestamps,
                uncertainty=uncertainty,
                horizon=horizon,
                model_name="lag-llama",
                metadata={
                    "context_length": ctx_len,
                    "num_samples": 100
                }
            )

        except Exception as e:
            logger.error(f"Lag-Llama forecast failed: {e}")
            return ForecastResult(success=False, error=str(e))
