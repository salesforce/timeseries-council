# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for Moirai2 anomaly detector.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from timeseries_council.detectors.moirai import MoiraiAnomalyDetector
from timeseries_council.types import DetectionResult


@pytest.fixture
def sample_series():
    """Create a sample time series with some anomalies."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    values = np.random.normal(50, 5, 200)
    
    # Inject anomalies
    values[50] = 150  # Extreme spike
    values[100] = 10   # Drop
    values[150] = 200  # Another extreme spike
    
    return pd.Series(values, index=dates)


@pytest.fixture
def short_series():
    """Create a series too short for Moirai2 detection."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    values = np.random.normal(50, 5, 30)
    return pd.Series(values, index=dates)


class TestMoiraiAnomalyDetectorInit:
    """Tests for detector initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        detector = MoiraiAnomalyDetector()
        
        assert detector.model_size == "small"
        assert detector.device == "cpu"
        assert detector.context_length == 64
        assert detector.stride == 1
        assert detector.confidence == 95.0
        assert detector.num_samples == 100
        
    def test_confidence_to_percentile_95(self):
        """Test 95% confidence converts to 2.5-97.5 percentiles."""
        detector = MoiraiAnomalyDetector(confidence=95.0)
        
        assert detector.lower_percentile == 2.5
        assert detector.upper_percentile == 97.5
        
    def test_confidence_to_percentile_99(self):
        """Test 99% confidence converts to 0.5-99.5 percentiles."""
        detector = MoiraiAnomalyDetector(confidence=99.0)
        
        assert detector.lower_percentile == 0.5
        assert detector.upper_percentile == 99.5
        
    def test_confidence_to_percentile_90(self):
        """Test 90% confidence converts to 5-95 percentiles."""
        detector = MoiraiAnomalyDetector(confidence=90.0)
        
        assert detector.lower_percentile == 5.0
        assert detector.upper_percentile == 95.0
        
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        detector = MoiraiAnomalyDetector(
            model_size="large",
            device="cuda",
            context_length=128,
            stride=5,
            confidence=99.0,
            num_samples=200
        )
        
        assert detector.model_size == "large"
        assert detector.device == "cuda"
        assert detector.context_length == 128
        assert detector.stride == 5
        assert detector.confidence == 99.0
        assert detector.num_samples == 200
        
    def test_name_property(self):
        """Test name property includes model size."""
        detector = MoiraiAnomalyDetector(model_size="base")
        assert "base" in detector.name.lower()
        
    def test_stride_minimum(self):
        """Test stride is at least 1."""
        detector = MoiraiAnomalyDetector(stride=0)
        assert detector.stride == 1
        
        detector = MoiraiAnomalyDetector(stride=-5)
        assert detector.stride == 1


class TestSeverityComputation:
    """Tests for severity calculation."""
    
    def test_compute_severity_outside_upper(self):
        """Test severity when value is above upper bound."""
        detector = MoiraiAnomalyDetector()
        
        # Value 150, bounds 40-60, std=10
        # Distance = 150 - 60 = 90, severity = 90 / 10 = 9.0
        severity, label = detector._compute_severity(150, 40, 60, 10)
        
        assert severity == 9.0
        assert label == "extreme"
        
    def test_compute_severity_outside_lower(self):
        """Test severity when value is below lower bound."""
        detector = MoiraiAnomalyDetector()
        
        # Value 10, bounds 40-60, std=10
        # Distance = 40 - 10 = 30, severity = 30 / 10 = 3.0
        severity, label = detector._compute_severity(10, 40, 60, 10)
        
        assert severity == 3.0
        assert label == "severe"
        
    def test_compute_severity_extreme_outlier(self):
        """Test severity for extreme outliers (e.g., 100000 when range is 0-100)."""
        detector = MoiraiAnomalyDetector()
        
        # Value 100000, bounds 0-100, std=25
        # Distance = 100000 - 100 = 99900, severity = 99900 / 25 = 3996.0
        severity, label = detector._compute_severity(100000, 0, 100, 25)
        
        assert severity > 100  # Very high severity
        assert label == "extreme"
        
    def test_compute_severity_moderate(self):
        """Test moderate severity classification (1-2 std devs)."""
        detector = MoiraiAnomalyDetector()
        
        # 1.5 std devs outside
        severity, label = detector._compute_severity(75, 40, 60, 10)  # 15/10 = 1.5
        
        assert 1.0 <= severity < 2.0
        assert label == "moderate"
        
    def test_compute_severity_normal(self):
        """Test normal value (within bounds)."""
        detector = MoiraiAnomalyDetector()
        
        severity, label = detector._compute_severity(50, 40, 60, 10)
        
        assert severity == 0.0
        assert label == "normal"
        
    def test_compute_severity_zero_std(self):
        """Test handling of zero standard deviation."""
        detector = MoiraiAnomalyDetector()
        
        # Should not raise division by zero
        severity, label = detector._compute_severity(100, 40, 60, 0)
        
        assert severity > 0
        assert label in ["moderate", "severe", "extreme"]


class TestValidation:
    """Tests for input validation."""
    
    def test_empty_series(self):
        """Test detection with empty series."""
        detector = MoiraiAnomalyDetector()
        empty_series = pd.Series([], dtype=float)
        
        result = detector.detect(empty_series)
        
        assert result.success is False
        assert "empty" in result.error.lower()
        
    def test_series_too_short(self, short_series):
        """Test detection with series shorter than context_length."""
        detector = MoiraiAnomalyDetector(context_length=64)
        
        # Series has 30 points but we need > 64
        result = detector.detect(short_series)
        
        assert result.success is False
        assert "more than" in result.error.lower() or "64" in result.error


class TestDetectWithMockedModel:
    """Tests for detect method with mocked Moirai2 model."""
    
    @patch.object(MoiraiAnomalyDetector, '_load_model')
    @patch.object(MoiraiAnomalyDetector, '_predict_point')
    def test_detect_finds_spike(self, mock_predict, mock_load, sample_series):
        """Test that detector identifies spikes."""
        mock_load.return_value = True
        
        # Mock predictions to return normal range
        def predict_side_effect(context, timestamp):
            return {
                "samples": np.random.normal(50, 5, 100),
                "lower": 35,
                "upper": 65,
                "median": 50,
                "std": 5
            }
        mock_predict.side_effect = predict_side_effect
        
        detector = MoiraiAnomalyDetector(context_length=64)
        result = detector.detect(sample_series)
        
        assert result.success is True
        # Should find the injected anomalies (150, 10, 200)
        assert result.anomaly_count > 0
        
    @patch.object(MoiraiAnomalyDetector, '_load_model')
    def test_detect_model_load_failure(self, mock_load, sample_series):
        """Test handling of model load failure."""
        mock_load.return_value = False
        
        detector = MoiraiAnomalyDetector()
        result = detector.detect(sample_series)
        
        assert result.success is False
        assert "Failed to load" in result.error


class TestDetectorFactory:
    """Tests for detector factory integration."""
    
    def test_create_moirai_detector(self):
        """Test creating detector via factory."""
        from timeseries_council.detectors.factory import create_detector
        
        detector = create_detector("moirai", auto_setup=False)
        
        assert detector is not None
        assert isinstance(detector, MoiraiAnomalyDetector)
        
    def test_moirai_in_available_detectors(self):
        """Test that moirai appears in available detectors list."""
        from timeseries_council.detectors.factory import get_available_detectors
        
        available = get_available_detectors()
        
        assert "moirai" in available
