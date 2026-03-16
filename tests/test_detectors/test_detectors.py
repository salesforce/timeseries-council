# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for detector implementations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from timeseries_council.detectors.base import BaseDetector, EnsembleDetector
from timeseries_council.detectors.factory import create_detector, list_detectors
from timeseries_council.detectors.zscore import ZScoreDetector
from timeseries_council.detectors.mad import MADDetector
from timeseries_council.types import DetectionResult


class TestBaseDetector:
    """Tests for BaseDetector abstract class."""

    def test_cannot_instantiate_abstract(self):
        """BaseDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector()

    def test_subclass_must_implement_methods(self):
        """Subclass must implement abstract methods."""
        class IncompleteDetector(BaseDetector):
            pass

        with pytest.raises(TypeError):
            IncompleteDetector()


class TestZScoreDetector:
    """Tests for Z-score detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = ZScoreDetector()
        assert detector.name == "zscore"

    def test_detect_returns_result(self, sample_series_with_anomalies):
        """Test that detect returns a DetectionResult."""
        detector = ZScoreDetector()
        result = detector.detect(sample_series_with_anomalies, sensitivity=2.0)

        assert isinstance(result, DetectionResult)
        assert result.success is True
        assert isinstance(result.anomalies, list)
        assert isinstance(result.scores, list)

    def test_detect_finds_anomalies(self, sample_series_with_anomalies):
        """Test that detector finds injected anomalies."""
        detector = ZScoreDetector()
        result = detector.detect(sample_series_with_anomalies, sensitivity=2.0)

        # Should find at least one anomaly
        assert len(result.anomalies) > 0

    def test_detect_with_progress_callback(self, sample_series_with_anomalies, progress_callback):
        """Test detect with progress callback."""
        detector = ZScoreDetector()
        result = detector.detect(
            sample_series_with_anomalies,
            sensitivity=2.0,
            progress_callback=progress_callback
        )

        assert result.success is True

    def test_sensitivity_affects_detection(self, sample_series_with_anomalies):
        """Test that sensitivity parameter affects detection count."""
        detector = ZScoreDetector()

        # Low sensitivity (high threshold) = fewer anomalies
        result_low = detector.detect(sample_series_with_anomalies, sensitivity=4.0)

        # High sensitivity (low threshold) = more anomalies
        result_high = detector.detect(sample_series_with_anomalies, sensitivity=1.5)

        # Higher sensitivity should find more anomalies
        assert len(result_high.anomalies) >= len(result_low.anomalies)


class TestMADDetector:
    """Tests for MAD detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MADDetector()
        assert detector.name == "mad"

    def test_detect_returns_result(self, sample_series_with_anomalies):
        """Test that detect returns a DetectionResult."""
        detector = MADDetector()
        result = detector.detect(sample_series_with_anomalies, sensitivity=3.0)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_robust_to_outliers(self, sample_series_with_anomalies):
        """Test that MAD is robust to outliers."""
        detector = MADDetector()
        result = detector.detect(sample_series_with_anomalies, sensitivity=3.0)

        # MAD should detect the injected anomalies
        assert len(result.anomalies) > 0


class TestDetectorFactory:
    """Tests for detector factory."""

    def test_list_detectors(self):
        """Test listing available detectors."""
        detectors = list_detectors()
        assert isinstance(detectors, list)
        assert "zscore" in detectors
        assert "mad" in detectors

    def test_create_zscore(self):
        """Test creating zscore detector."""
        detector = create_detector("zscore")
        assert detector is not None
        assert detector.name == "zscore"

    def test_create_mad(self):
        """Test creating MAD detector."""
        detector = create_detector("mad")
        assert detector is not None
        assert detector.name == "mad"

    def test_create_unknown_detector(self):
        """Test creating unknown detector returns None."""
        detector = create_detector("nonexistent_detector")
        assert detector is None


class TestEnsembleDetector:
    """Tests for ensemble detector."""

    def test_initialization(self, mock_detector):
        """Test ensemble initialization."""
        ensemble = EnsembleDetector([mock_detector])
        assert ensemble.name == "ensemble"
        assert len(ensemble.detectors) == 1

    def test_ensemble_detect(self, sample_series_with_anomalies, mock_detector):
        """Test ensemble detection."""
        ensemble = EnsembleDetector([mock_detector, mock_detector])
        result = ensemble.detect(sample_series_with_anomalies, sensitivity=2.0)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_empty_ensemble_fails(self, sample_series_with_anomalies):
        """Test that empty ensemble handles gracefully."""
        ensemble = EnsembleDetector([])
        result = ensemble.detect(sample_series_with_anomalies, sensitivity=2.0)

        assert result.success is False


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_to_dict(self):
        """Test DetectionResult to_dict method."""
        result = DetectionResult(
            success=True,
            anomalies=[{"index": 0, "value": 100.0}],
            scores=[1.0, 2.0, 3.0],
            threshold=2.0,
            detector_name="test"
        )

        d = result.to_dict()
        assert d["success"] is True
        assert len(d["anomalies"]) == 1
        assert d["detector_name"] == "test"

    def test_failed_result(self):
        """Test failed DetectionResult."""
        result = DetectionResult(
            success=False,
            error="Detector failed"
        )

        assert result.success is False
        assert result.error == "Detector failed"
        assert result.anomalies is None
