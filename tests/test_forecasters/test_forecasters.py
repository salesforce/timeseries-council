# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for forecaster implementations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from timeseries_council.forecasters.base import BaseForecaster, EnsembleForecaster
from timeseries_council.forecasters.factory import create_forecaster, list_forecasters
from timeseries_council.forecasters.zscore_baseline import ZScoreBaselineForecaster
from timeseries_council.types import ForecastResult


class TestBaseForecaster:
    """Tests for BaseForecaster abstract class."""

    def test_cannot_instantiate_abstract(self):
        """BaseForecaster cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseForecaster()

    def test_subclass_must_implement_methods(self):
        """Subclass must implement abstract methods."""
        class IncompleteForecaster(BaseForecaster):
            pass

        with pytest.raises(TypeError):
            IncompleteForecaster()


class TestZScoreBaselineForecaster:
    """Tests for Z-score baseline forecaster."""

    def test_initialization(self):
        """Test forecaster initialization."""
        forecaster = ZScoreBaselineForecaster()
        assert forecaster.name == "zscore_baseline"

    def test_forecast_returns_result(self, sample_series):
        """Test that forecast returns a ForecastResult."""
        forecaster = ZScoreBaselineForecaster()
        result = forecaster.forecast(sample_series, horizon=5)

        assert isinstance(result, ForecastResult)
        assert result.success is True
        assert len(result.forecast) == 5
        assert len(result.dates) == 5

    def test_forecast_with_progress_callback(self, sample_series, progress_callback):
        """Test forecast with progress callback."""
        forecaster = ZScoreBaselineForecaster()
        result = forecaster.forecast(sample_series, horizon=5, progress_callback=progress_callback)

        assert result.success is True
        assert len(progress_callback.calls) > 0


class TestForecasterFactory:
    """Tests for forecaster factory."""

    def test_list_forecasters(self):
        """Test listing available forecasters."""
        forecasters = list_forecasters()
        assert isinstance(forecasters, list)
        assert "zscore_baseline" in forecasters

    def test_create_zscore_baseline(self):
        """Test creating zscore baseline forecaster."""
        forecaster = create_forecaster("zscore_baseline")
        assert forecaster is not None
        assert forecaster.name == "zscore_baseline"

    def test_create_unknown_forecaster(self):
        """Test creating unknown forecaster returns None."""
        forecaster = create_forecaster("nonexistent_forecaster")
        assert forecaster is None

    def test_create_moirai_without_merlion(self):
        """Test moirai2 creation handles missing dependencies."""
        with patch.dict('sys.modules', {'merlion': None}):
            forecaster = create_forecaster("moirai")
            # Should return None or raise if merlion not available
            # Behavior depends on implementation


class TestEnsembleForecaster:
    """Tests for ensemble forecaster."""

    def test_initialization(self, mock_forecaster):
        """Test ensemble initialization."""
        ensemble = EnsembleForecaster([mock_forecaster])
        assert ensemble.name == "ensemble"
        assert len(ensemble.forecasters) == 1

    def test_ensemble_forecast(self, sample_series, mock_forecaster):
        """Test ensemble forecasting."""
        ensemble = EnsembleForecaster([mock_forecaster, mock_forecaster])
        result = ensemble.forecast(sample_series, horizon=5)

        assert isinstance(result, ForecastResult)
        assert result.success is True

    def test_empty_ensemble_fails(self, sample_series):
        """Test that empty ensemble handles gracefully."""
        ensemble = EnsembleForecaster([])
        result = ensemble.forecast(sample_series, horizon=5)

        assert result.success is False


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    def test_to_dict(self):
        """Test ForecastResult to_dict method."""
        result = ForecastResult(
            success=True,
            forecast=[100.0, 101.0],
            dates=["2024-01-01", "2024-01-02"],
            model_name="test"
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["forecast"] == [100.0, 101.0]
        assert d["model_name"] == "test"

    def test_failed_result(self):
        """Test failed ForecastResult."""
        result = ForecastResult(
            success=False,
            error="Model failed to load"
        )

        assert result.success is False
        assert result.error == "Model failed to load"
        assert result.forecast is None
