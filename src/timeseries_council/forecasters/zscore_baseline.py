# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Simple baseline forecaster using statistical methods.
No external dependencies - always works.
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable, List
from datetime import timedelta

from .base import BaseForecaster
from ..types import ForecastResult, ProgressStage
from ..logging import get_logger

logger = get_logger(__name__)


class ZScoreBaselineForecaster(BaseForecaster):
    """
    Simple baseline forecaster using rolling statistics.

    Uses a combination of:
    - Rolling mean for trend
    - Rolling std for confidence intervals

    This forecaster has no external dependencies and is useful for:
    - Testing and development
    - Baseline comparison
    - Fallback when other models fail
    """

    def __init__(self, window: int = 14):
        """
        Initialize the baseline forecaster.

        Args:
            window: Rolling window size for statistics
        """
        self.window = window
        logger.info(f"Initialized ZScoreBaselineForecaster with window={window}")

    @property
    def name(self) -> str:
        return "zscore_baseline"

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressStage, str, float], None]] = None
    ) -> ForecastResult:
        """
        Generate forecast using simple statistical methods.

        Args:
            series: Historical time series data
            horizon: Number of periods to forecast
            context_length: Not used (for API compatibility)
            progress_callback: Optional progress callback

        Returns:
            ForecastResult with predictions and confidence intervals
        """
        try:
            if progress_callback:
                progress_callback(ProgressStage.FORECASTING, "Starting baseline forecast...", 0.1)

            # Ensure numeric
            series = pd.to_numeric(series, errors='coerce').dropna()

            if len(series) < self.window:
                return ForecastResult(
                    success=False,
                    error=f"Need at least {self.window} data points, got {len(series)}"
                )

            if progress_callback:
                progress_callback(ProgressStage.FORECASTING, "Computing statistics...", 0.3)

            # Calculate rolling statistics
            rolling_mean = series.rolling(window=self.window).mean()
            rolling_std = series.rolling(window=self.window).std()

            # Use last valid values
            last_mean = rolling_mean.iloc[-1]
            last_std = rolling_std.iloc[-1]

            # Calculate trend (simple linear)
            recent = series.tail(self.window)
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent.values, 1)[0]

            if progress_callback:
                progress_callback(ProgressStage.FORECASTING, "Generating predictions...", 0.6)

            # Generate forecasts
            forecast_values = []
            confidence_lower = []
            confidence_upper = []

            for i in range(1, horizon + 1):
                # Trend-adjusted prediction
                pred = last_mean + (slope * i)
                forecast_values.append(float(pred))

                # Confidence intervals (widening with horizon)
                uncertainty = last_std * np.sqrt(i) * 1.96
                confidence_lower.append(float(pred - uncertainty))
                confidence_upper.append(float(pred + uncertainty))

            # Generate dates - use robust frequency inference
            if isinstance(series.index, pd.DatetimeIndex):
                from ..utils import infer_frequency
                last_date = series.index[-1]
                freq = infer_frequency(series)
                dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
                date_strings = [d.strftime('%Y-%m-%d %H:%M:%S') if freq in ['T', 'min', 'H', 'S'] or 'T' in freq or 'H' in freq else d.strftime('%Y-%m-%d') for d in dates]
            else:
                date_strings = [f"t+{i}" for i in range(1, horizon + 1)]

            if progress_callback:
                progress_callback(ProgressStage.FORECASTING, "Forecast complete", 1.0)

            logger.info(f"Baseline forecast generated: {horizon} periods")

            return ForecastResult(
                success=True,
                forecast=forecast_values,
                timestamps=date_strings,
                lower_bound=confidence_lower,
                upper_bound=confidence_upper,
                model_name=self.name,
                horizon=horizon,
                metadata={
                    "method": "rolling_statistics",
                    "window": self.window,
                    "trend_slope": float(slope)
                }
            )

        except Exception as e:
            logger.error(f"Baseline forecast failed: {e}")
            return ForecastResult(
                success=False,
                error=str(e),
                model_name=self.name
            )
