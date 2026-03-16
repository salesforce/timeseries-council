# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Amazon Chronos2 forecaster.
"""

from typing import Optional, Callable
import pandas as pd

from .base import BaseForecaster
from ..types import ForecastResult
from ..logging import get_logger
from ..exceptions import ForecasterError
from ..utils import get_device

logger = get_logger(__name__)


class ChronosForecaster(BaseForecaster):
    """Amazon Chronos2 foundation model forecaster."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = None
    ):
        """
        Initialize Chronos2 forecaster.

        Args:
            model_size: Model size ('base', 'synth', 'small')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_size = model_size
        self.device = get_device(device)
        self._pipeline = None

        # Model name mapping for Chronos2
        self._model_names = {
            "base": "amazon/chronos-2",
            "synth": "autogluon/chronos-2-synth",
            "small": "autogluon/chronos-2-small",
        }

        logger.info(f"Initialized Chronos2Forecaster: {model_size} on {self.device}")

    @property
    def name(self) -> str:
        return f"Chronos-{self.model_size}"

    @property
    def description(self) -> str:
        return f"Amazon Chronos foundation model ({self.model_size})"

    def _get_pipeline(self):
        """Lazy load the Chronos2 pipeline."""
        if self._pipeline is None:
            try:
                from chronos import Chronos2Pipeline
                import torch

                model_name = self._model_names.get(self.model_size, self._model_names["base"])
                device_map = self.device if self.device != "cpu" else "cpu"
                dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

                logger.info(f"Loading Chronos2 model: {model_name}")
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    model_name,
                    device_map=device_map,
                    dtype=dtype,  # Use dtype instead of deprecated torch_dtype
                )
            except ImportError:
                raise ForecasterError(
                    "chronos-forecasting not installed. Run: pip install chronos-forecasting",
                    {"forecaster": "chronos2"}
                )
        return self._pipeline

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using Chronos2 model."""
        import numpy as np

        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading Chronos2 model...", 0.1)

        try:
            import torch
            pipeline = self._get_pipeline()
        except (ImportError, ForecasterError) as e:
            return ForecastResult(success=False, error=str(e))

        try:
            self._report_progress(progress_callback, "Preparing data...", 0.3)

            # Prepare context - Chronos2 expects DataFrame format
            if context_length:
                context = series.tail(context_length)
            else:
                context = series

            # Ensure index is datetime
            if not pd.api.types.is_datetime64_any_dtype(context.index):
                context.index = pd.to_datetime(context.index)

            # Handle duplicate timestamps by taking the last value
            if not context.index.is_unique:
                context = context[~context.index.duplicated(keep='last')]

            # Try to infer frequency
            freq = pd.infer_freq(context.index)
            
            # If frequency cannot be inferred, try to estimate minimal diff
            if freq is None and len(context) > 1:
                # Calculate time differences
                diffs = context.index.to_series().diff().dropna()
                # Use the mode of differences as the likely frequency
                if not diffs.empty:
                    likely_freq = diffs.mode().iloc[0]
                    # Create a regular index with this frequency
                    try:
                        # Reindex to regular frequency, filling missing values with NaN (Chronos handles NaNs)
                        context = context.asfreq(likely_freq)
                        freq = likely_freq
                    except Exception:
                        pass # Keep original if reindexing fails

            # Convert to DataFrame format expected by Chronos2
            # Chronos2 requires: timestamp, target, and item_id columns
            context_df = pd.DataFrame({
                'timestamp': context.index,
                'target': context.values,
                'item_id': 'series_0'  # Required by Chronos2 for identifying time series
            })

            self._report_progress(progress_callback, "Generating forecast...", 0.5)

            # Generate forecast using Chronos2's predict_df method
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],  # For uncertainty estimation
                id_column='item_id',  # Specify the ID column name
                timestamp_column='timestamp',
                target='target'
            )

            self._report_progress(progress_callback, "Processing results...", 0.8)

            # Extract median forecast (0.5 quantile) and uncertainty
            median_forecast = pred_df['0.5'].values
            # Calculate uncertainty from quantile range
            uncertainty = (pred_df['0.9'].values - pred_df['0.1'].values) / 2.0

            # Extract timestamps from prediction DataFrame
            timestamps = [str(t) for t in pred_df['timestamp'].values]

            self._report_progress(progress_callback, "Forecast complete", 1.0)

            logger.info(f"Chronos2 forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=median_forecast.round(2).tolist(),
                timestamps=timestamps,
                uncertainty=uncertainty.round(2).tolist(),
                horizon=horizon,
                model_name=f"chronos2-{self.model_size}",
                metadata={
                    "context_length": len(context),
                    "quantile_levels": [0.1, 0.5, 0.9]
                }
            )

        except Exception as e:
            logger.error(f"Chronos2 forecast failed: {e}")
            return ForecastResult(success=False, error=str(e))

