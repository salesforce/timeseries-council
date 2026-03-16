# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Moirai forecaster using uni2ts library (official HuggingFace implementation).
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger
from ..utils import get_device

logger = get_logger(__name__)

# Model ID mapping
MOIRAI_MODELS = {
    "small": "Salesforce/moirai-2.0-R-small",
    "base": "Salesforce/moirai-2.0-R-base",
    "large": "Salesforce/moirai-2.0-R-large",
}


class MoiraiForecaster(BaseForecaster):
    """Moirai foundation model forecaster using uni2ts."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = None
    ):
        """
        Initialize Moirai forecaster.

        Args:
            model_size: Model size ('small', 'base', 'large')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_size = model_size.lower()
        self.device = get_device(device)
        self._model = None
        self._pipeline = None
        logger.info(f"Initialized MoiraiForecaster: {self.model_size} on {self.device}")

    @property
    def name(self) -> str:
        return f"Moirai-{self.model_size}"

    @property
    def description(self) -> str:
        return f"Salesforce Moirai foundation model ({self.model_size})"

    def _get_model_id(self) -> str:
        """Get HuggingFace model ID."""
        return MOIRAI_MODELS.get(self.model_size, MOIRAI_MODELS["small"])

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using Moirai model."""
        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading Moirai model...", 0.1)

        try:
            import torch
            from einops import rearrange
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            return ForecastResult(
                success=False,
                error="Required packages not installed. Run: pip install torch einops huggingface_hub"
            )

        try:
            # Try to import uni2ts for official Moirai2 support
            try:
                from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
                USE_UNI2TS = True
            except ImportError:
                USE_UNI2TS = False
                logger.warning("uni2ts not available, using fallback implementation")

            # Prepare data
            ctx_len = context_length or min(512, len(series))
            context = series.tail(ctx_len).values.astype(np.float32)

            self._report_progress(progress_callback, "Preparing forecast...", 0.3)

            if USE_UNI2TS:
                # Use official uni2ts implementation with GluonTS predictor pattern
                from gluonts.dataset.pandas import PandasDataset

                model_id = self._get_model_id()
                logger.info(f"Loading Moirai2 from: {model_id}")

                # Load Moirai2Module from HuggingFace (handles caching automatically)
                module = Moirai2Module.from_pretrained(model_id)
                if module is None:
                    raise ValueError("Failed to load Moirai2Module from HuggingFace")

                # Create Moirai2Forecast wrapper with the loaded module
                forecast_model = Moirai2Forecast(
                    module=module,
                    prediction_length=horizon,
                    context_length=ctx_len,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )

                self._report_progress(progress_callback, "Creating predictor...", 0.5)

                # Create predictor
                predictor = forecast_model.create_predictor(batch_size=1, device=torch.device(self.device))
                if predictor is None:
                    raise ValueError("Failed to create Moirai predictor")

                self._report_progress(progress_callback, "Generating forecast...", 0.6)

                # Prepare GluonTS dataset - must include timestamp column
                context_series = series.tail(ctx_len).copy()

                # Log input data characteristics
                logger.info(f"Input series: len={len(series)}, dtype={series.dtype}, index_type={type(series.index).__name__}")

                # Ensure we have a DatetimeIndex
                if not isinstance(context_series.index, pd.DatetimeIndex):
                    logger.warning("Converting series index to DatetimeIndex")
                    context_series.index = pd.to_datetime(context_series.index)

                # Ensure data is clean
                context_series = context_series.dropna()
                if len(context_series) < 10:
                    raise ValueError(f"Insufficient data after cleaning: {len(context_series)} points")

                # Convert to float32 for PyTorch compatibility
                target_values = context_series.values.astype(np.float32)

                # Create dataframe with proper structure for GluonTS
                # GluonTS expects univariate time series in wide format for from_long_dataframe
                df = pd.DataFrame({
                    "timestamp": context_series.index,
                    "target": target_values,
                    "item_id": "series"
                })

                logger.info(f"Dataset df: shape={df.shape}, dtypes={df.dtypes.to_dict()}")

                # Use dict-based dataset creation which is more reliable
                from gluonts.dataset.common import ListDataset
                
                # Robust frequency inference for GluonTS
                inferred_freq = pd.infer_freq(context_series.index)
                if inferred_freq is None:
                    # Fallback: estimate from median time delta
                    time_diffs = pd.Series(context_series.index).diff().dropna()
                    if len(time_diffs) > 0:
                        median_diff = time_diffs.median()
                        # Map to pandas frequency strings
                        if median_diff >= pd.Timedelta(days=28):
                            inferred_freq = "MS"  # Month start
                        elif median_diff >= pd.Timedelta(days=7):
                            inferred_freq = "W"
                        elif median_diff >= pd.Timedelta(days=1):
                            inferred_freq = "D"
                        elif median_diff >= pd.Timedelta(hours=1):
                            inferred_freq = "h"
                        else:
                            inferred_freq = "min"
                    else:
                        inferred_freq = "D"
                
                logger.info(f"Using frequency: {inferred_freq}")
                
                dataset = ListDataset(
                    [{"start": context_series.index[0], "target": target_values}],
                    freq=inferred_freq
                )

                # Generate forecasts - wrap in try-except for internal Moirai2 errors
                try:
                    forecasts = list(predictor.predict(dataset))
                except Exception as pred_err:
                    logger.warning(f"Moirai2 predict failed ({pred_err}), using statistical fallback")
                    USE_UNI2TS = False  # Force fallback
                    forecasts = []

                if forecasts:
                    forecast_obj = forecasts[0]

                    # Get median forecast and uncertainty - with defensive handling
                    try:
                        forecast_values = forecast_obj.median.tolist()
                    except Exception as quant_err:
                        logger.warning(f"Failed to get median, using mean: {quant_err}")
                        forecast_values = forecast_obj.mean.tolist()
                    
                    try:
                        q10 = forecast_obj.quantile(0.1)
                        q90 = forecast_obj.quantile(0.9)
                        uncertainty = ((q90 - q10) / 2).tolist()
                    except Exception as quant_err:
                        logger.warning(f"Failed to get quantiles, using std estimate: {quant_err}")
                        # Fallback: estimate uncertainty from historical std
                        historical_std = float(np.std(target_values))
                        uncertainty = [historical_std] * len(forecast_values)
                else:
                    USE_UNI2TS = False  # Force fallback below
            
            if not USE_UNI2TS:
                # Fallback: use a simple statistical forecast if uni2ts not available
                logger.warning("Using statistical fallback for Moirai")

                # Simple exponential smoothing as fallback
                from scipy.ndimage import uniform_filter1d
                smoothed = uniform_filter1d(context, size=min(7, len(context)))

                # Trend estimation
                if len(context) >= 7:
                    trend = (smoothed[-1] - smoothed[-7]) / 7
                else:
                    trend = 0

                # Generate forecast with trend
                last_val = context[-1]
                forecast_values = [round(last_val + trend * (i + 1), 2) for i in range(horizon)]
                uncertainty = [round(np.std(context[-14:]) if len(context) >= 14 else np.std(context), 2)] * horizon

            self._report_progress(progress_callback, "Processing results...", 0.8)

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

            logger.info(f"Moirai forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=[round(v, 2) for v in forecast_values],
                timestamps=timestamps,
                uncertainty=uncertainty,
                horizon=horizon,
                model_name=f"moirai-{self.model_size}",
                metadata={
                    "context_length": ctx_len,
                    "device": self.device,
                    "fallback_used": not USE_UNI2TS
                }
            )

        except Exception as e:
            logger.error(f"Moirai forecast failed: {e}")
            import traceback
            return ForecastResult(
                success=False,
                error=str(e),
                metadata={"traceback": traceback.format_exc()}
            )
