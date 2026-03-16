# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Pytest configuration and shared fixtures for timeseries-council tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_series():
    """Create a sample time series for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.Series(values, index=dates, name='test_values')


@pytest.fixture
def sample_series_with_anomalies():
    """Create a sample time series with known anomalies."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(100) * 2)
    # Inject anomalies at known positions
    values[25] = values[25] + 50  # Spike
    values[50] = values[50] - 40  # Drop
    values[75] = values[75] + 45  # Spike
    return pd.Series(values, index=dates, name='test_values')


@pytest.fixture
def sample_dataframe(sample_series):
    """Create a sample DataFrame for testing."""
    df = pd.DataFrame({'sales': sample_series.values}, index=sample_series.index)
    df['date'] = df.index
    return df


@pytest.fixture
def sample_csv_path(tmp_path, sample_dataframe):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.provider_name = "mock"
    provider.model_name = "mock-model"
    provider.generate.return_value = "This is a mock response."
    provider.generate_with_tools.return_value = {
        "tool_call": None,
        "response": "Mock analysis complete."
    }
    return provider


@pytest.fixture
def mock_forecaster():
    """Create a mock forecaster."""
    from timeseries_council.types import ForecastResult

    forecaster = MagicMock()
    forecaster.name = "mock_forecaster"
    forecaster.forecast.return_value = ForecastResult(
        success=True,
        forecast=[100.0, 101.0, 102.0, 103.0, 104.0],
        dates=["2024-04-10", "2024-04-11", "2024-04-12", "2024-04-13", "2024-04-14"],
        confidence_lower=[95.0, 96.0, 97.0, 98.0, 99.0],
        confidence_upper=[105.0, 106.0, 107.0, 108.0, 109.0],
        model_name="mock_forecaster"
    )
    return forecaster


@pytest.fixture
def mock_detector():
    """Create a mock detector."""
    from timeseries_council.types import DetectionResult

    detector = MagicMock()
    detector.name = "mock_detector"
    detector.detect.return_value = DetectionResult(
        success=True,
        anomalies=[
            {"index": 25, "date": "2024-01-26", "value": 150.0, "score": 3.5},
            {"index": 50, "date": "2024-02-20", "value": 60.0, "score": -3.2}
        ],
        anomaly_count=2,
        detector_name="mock_detector"
    )
    return detector


@pytest.fixture
def progress_callback():
    """Create a progress callback that records calls.
    
    Accepts both 2-arg (message, progress) and 3-arg (stage, message, progress) forms.
    """
    calls = []

    def callback(*args):
        calls.append(args)

    callback.calls = calls
    return callback
