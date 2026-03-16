# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Abstract base class for anomaly detection models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List
import pandas as pd

from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the detector name for display."""
        pass

    @property
    def description(self) -> str:
        """Return a brief description of the detector."""
        return f"{self.name} anomaly detector"

    @abstractmethod
    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """
        Detect anomalies in the given time series.

        Args:
            series: Pandas Series with DatetimeIndex containing data
            sensitivity: Sensitivity threshold (interpretation varies by detector)
            progress_callback: Optional callback for progress updates
                               Signature: (message: str, progress: float 0-1)
            memory: Optional detection memory with previous anomalies,
                    baseline stats, and domain context for informed detection

        Returns:
            DetectionResult with list of anomalies and metadata
        """
        pass

    def validate_input(self, series: pd.Series) -> Optional[str]:
        """
        Validate input data before detection.

        Args:
            series: Input time series

        Returns:
            Error message if validation fails, None if valid
        """
        if series is None or len(series) == 0:
            return "Input series is empty"

        if not isinstance(series.index, pd.DatetimeIndex):
            return "Series must have DatetimeIndex"

        if len(series) < 3:
            return "Need at least 3 data points for detection"

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

    def _classify_anomaly(self, value: float, mean: float, std: float) -> AnomalyType:
        """Classify anomaly as spike or drop based on value."""
        if value > mean + std:
            return AnomalyType.SPIKE
        elif value < mean - std:
            return AnomalyType.DROP
        else:
            return AnomalyType.SHIFT

    def _apply_memory(
        self,
        anomalies: List[Anomaly],
        memory: Optional[DetectionMemory],
    ) -> List[Anomaly]:
        """Apply memory context to detected anomalies (post-processing).

        Two operations:
        1. Baseline rescoring: if baseline_stats has mean/std, rescore each
           anomaly relative to the baseline (z-score from baseline).
        2. Expected-range filtering: if expected_range is [low, high], drop
           anomalies whose values fall inside that range.

        Note: For deep integration (using baseline_stats during core scoring),
        see ZScoreDetector and MADDetector which read memory.baseline_stats
        directly in their detect() methods.

        Args:
            anomalies: List of detected anomalies
            memory: Optional detection memory

        Returns:
            Adjusted list of anomalies
        """
        if memory is None or not anomalies:
            return anomalies

        baseline_mean = memory.baseline_stats.get("mean")
        baseline_std = memory.baseline_stats.get("std")
        expected_range = memory.expected_range

        adjusted = []
        for anomaly in anomalies:
            # Rescore relative to baseline if provided
            if baseline_mean is not None and baseline_std is not None and baseline_std > 0:
                anomaly.score = abs(anomaly.value - baseline_mean) / baseline_std

            # Filter out anomalies within the expected range
            if expected_range and len(expected_range) == 2:
                low, high = expected_range
                if low <= anomaly.value <= high:
                    continue

            adjusted.append(anomaly)

        return adjusted


class EnsembleDetector(BaseDetector):
    """Ensemble of multiple detectors with voting."""

    def __init__(
        self,
        detectors: List[BaseDetector],
        voting_threshold: float = 0.5
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of detectors to ensemble
            voting_threshold: Fraction of detectors that must agree (0-1)
        """
        if not detectors:
            raise ValueError("At least one detector required")

        self.detectors = detectors
        self.voting_threshold = voting_threshold
        logger.info(f"Created ensemble with {len(detectors)} detectors")

    @property
    def name(self) -> str:
        return "Ensemble"

    @property
    def description(self) -> str:
        names = [d.name for d in self.detectors]
        return f"Ensemble of: {', '.join(names)}"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Run all detectors and combine results via voting."""
        from collections import Counter

        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        # Track votes per timestamp
        votes: dict = {}
        anomaly_details: dict = {}

        for i, detector in enumerate(self.detectors):
            progress = i / len(self.detectors)
            self._report_progress(
                progress_callback,
                f"Running {detector.name}...",
                progress
            )

            try:
                result = detector.detect(series, sensitivity, memory=memory)
                if result.success and result.anomalies:
                    for anomaly in result.anomalies:
                        ts = anomaly.timestamp
                        if ts not in votes:
                            votes[ts] = 0
                            anomaly_details[ts] = anomaly
                        votes[ts] += 1
                else:
                    if not result.success:
                        logger.warning(f"{detector.name} failed: {result.error}")
            except Exception as e:
                logger.warning(f"{detector.name} exception: {e}")

        # Filter by voting threshold
        min_votes = int(len(self.detectors) * self.voting_threshold)
        min_votes = max(1, min_votes)

        final_anomalies = []
        for ts, vote_count in votes.items():
            if vote_count >= min_votes:
                anomaly = anomaly_details[ts]
                # Update score based on votes
                anomaly.score = vote_count / len(self.detectors)
                final_anomalies.append(anomaly)

        # Sort by timestamp
        final_anomalies.sort(key=lambda a: a.timestamp)

        # Apply memory to ensemble results
        final_anomalies = self._apply_memory(final_anomalies, memory)

        self._report_progress(progress_callback, "Ensemble complete", 1.0)

        # Compute stats
        mean_val = float(series.mean())
        std_val = float(series.std())

        return DetectionResult(
            success=True,
            anomaly_count=len(final_anomalies),
            anomalies=final_anomalies,
            detector_name=self.description,
            metadata={
                "detectors_used": len(self.detectors),
                "voting_threshold": self.voting_threshold,
                "min_votes_required": min_votes,
                "sensitivity": sensitivity,
                "mean": mean_val,
                "std": std_val,
                "memory_applied": memory is not None,
            }
        )
