# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Abstract base class for forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List
import pandas as pd

from ..types import ForecastResult
from ..logging import get_logger

logger = get_logger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the forecaster name for display."""
        pass

    @property
    def description(self) -> str:
        """Return a brief description of the forecaster."""
        return f"{self.name} forecaster"

    @abstractmethod
    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """
        Generate forecast for the given time series.

        Args:
            series: Pandas Series with DatetimeIndex containing historical data
            horizon: Number of steps to forecast
            context_length: Optional number of historical points to use
            progress_callback: Optional callback for progress updates
                               Signature: (message: str, progress: float 0-1)

        Returns:
            ForecastResult with forecast values, timestamps, and metadata
        """
        pass

    def validate_input(self, series: pd.Series, horizon: int) -> Optional[str]:
        """
        Validate input data before forecasting.

        Args:
            series: Input time series
            horizon: Forecast horizon

        Returns:
            Error message if validation fails, None if valid
        """
        if series is None or len(series) == 0:
            return "Input series is empty"

        if not isinstance(series.index, pd.DatetimeIndex):
            return "Series must have DatetimeIndex"

        if horizon <= 0:
            return "Horizon must be positive"

        if len(series) < 3:
            return "Need at least 3 data points for forecasting"

        return None

    def _report_progress(
        self,
        callback: Optional[Callable[[str, float], None]],
        message: str,
        progress: float
    ) -> None:
        """Helper to report progress if callback is provided."""
        if callback:
            callback(message, min(1.0, max(0.0, progress)))
        logger.debug(f"Progress: {progress:.0%} - {message}")


class EnsembleForecaster(BaseForecaster):
    """Ensemble of multiple forecasters with weighted averaging."""

    def __init__(
        self,
        forecasters: List[BaseForecaster],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble forecaster.

        Args:
            forecasters: List of forecasters to ensemble
            weights: Optional weights for each forecaster (default: equal)
        """
        if not forecasters:
            raise ValueError("At least one forecaster required")

        self.forecasters = forecasters
        self.weights = weights or [1.0 / len(forecasters)] * len(forecasters)

        if len(self.weights) != len(forecasters):
            raise ValueError("Number of weights must match number of forecasters")

        logger.info(f"Created ensemble with {len(forecasters)} forecasters")

    @property
    def name(self) -> str:
        return "Ensemble"

    @property
    def description(self) -> str:
        names = [f.name for f in self.forecasters]
        return f"Ensemble of: {', '.join(names)}"

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Run all forecasters and combine results."""
        import numpy as np

        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        all_forecasts = []
        all_uncertainties = []
        timestamps = None

        for i, (forecaster, weight) in enumerate(zip(self.forecasters, self.weights)):
            progress = i / len(self.forecasters)
            self._report_progress(
                progress_callback,
                f"Running {forecaster.name}...",
                progress
            )

            try:
                result = forecaster.forecast(series, horizon, context_length)
                if result.success and result.forecast:
                    all_forecasts.append((result.forecast, weight))
                    if result.uncertainty:
                        all_uncertainties.append((result.uncertainty, weight))
                    if timestamps is None:
                        timestamps = result.timestamps
                else:
                    logger.warning(f"{forecaster.name} failed: {result.error}")
            except Exception as e:
                logger.warning(f"{forecaster.name} exception: {e}")

        if not all_forecasts:
            return ForecastResult(
                success=False,
                error="All forecasters failed"
            )

        # Weighted average
        total_weight = sum(w for _, w in all_forecasts)
        ensemble_forecast = [0.0] * horizon

        for forecast, weight in all_forecasts:
            for i, val in enumerate(forecast[:horizon]):
                ensemble_forecast[i] += val * weight / total_weight

        # Combine uncertainties
        ensemble_uncertainty = None
        if all_uncertainties:
            total_unc_weight = sum(w for _, w in all_uncertainties)
            ensemble_uncertainty = [0.0] * horizon
            for unc, weight in all_uncertainties:
                for i, val in enumerate(unc[:horizon]):
                    ensemble_uncertainty[i] += val * weight / total_unc_weight

        self._report_progress(progress_callback, "Ensemble complete", 1.0)

        return ForecastResult(
            success=True,
            forecast=ensemble_forecast,
            timestamps=timestamps,
            uncertainty=ensemble_uncertainty,
            horizon=horizon,
            model_name=self.description,
            metadata={"forecasters_used": len(all_forecasts)}
        )
