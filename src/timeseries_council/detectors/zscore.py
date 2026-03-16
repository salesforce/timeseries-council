# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Z-score based anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)


class ZScoreDetector(BaseDetector):
    """Simple Z-score based anomaly detector."""

    def __init__(self, window: Optional[int] = None):
        """
        Initialize Z-score detector.

        Args:
            window: Optional rolling window size for local statistics.
                    If None, uses global statistics.
        """
        self.window = window
        logger.info(f"Initialized ZScoreDetector (window={window})")

    @property
    def name(self) -> str:
        if self.window:
            return f"Z-Score-{self.window}"
        return "Z-Score"

    @property
    def description(self) -> str:
        if self.window:
            return f"Z-score detector with rolling window of {self.window}"
        return "Z-score detector using global statistics"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using Z-score method."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Calculating statistics...", 0.2)

        try:
            # Deep integration: use baseline_stats from memory when available.
            # This means the z-score is computed relative to the baseline
            # distribution, so the detector scores points based on how far
            # they deviate from the *known normal*, not just the current batch.
            use_baseline = (
                memory is not None
                and memory.baseline_stats.get("mean") is not None
                and memory.baseline_stats.get("std") is not None
                and memory.baseline_stats["std"] > 0
            )

            if use_baseline:
                baseline_mean = memory.baseline_stats["mean"]
                baseline_std = memory.baseline_stats["std"]
                logger.info(
                    f"Using baseline stats for Z-score: mean={baseline_mean}, std={baseline_std}"
                )
                z_scores = (series - baseline_mean) / baseline_std
            elif self.window:
                # Rolling statistics
                rolling_mean = series.rolling(window=self.window, center=True).mean()
                rolling_std = series.rolling(window=self.window, center=True).std()
                # Fill edges with global stats
                global_mean = series.mean()
                global_std = series.std()
                rolling_mean = rolling_mean.fillna(global_mean)
                rolling_std = rolling_std.fillna(global_std)
                z_scores = (series - rolling_mean) / rolling_std
            else:
                # Global statistics
                mean = series.mean()
                std = series.std()
                z_scores = (series - mean) / std

            self._report_progress(progress_callback, "Finding anomalies...", 0.6)

            # Find anomalies
            anomaly_mask = np.abs(z_scores) > sensitivity
            anomaly_indices = series[anomaly_mask].index

            anomalies = []
            for idx in anomaly_indices:
                val = float(series[idx])
                z = float(z_scores[idx])
                anomaly_type = AnomalyType.SPIKE if z > 0 else AnomalyType.DROP

                anomalies.append(Anomaly(
                    timestamp=str(idx),
                    value=val,
                    score=abs(z),
                    anomaly_type=anomaly_type
                ))

            # Apply memory post-processing (expected_range filtering)
            anomalies = self._apply_memory(anomalies, memory)

            self._report_progress(progress_callback, "Detection complete", 1.0)

            mean_val = float(series.mean())
            std_val = float(series.std())

            logger.info(f"Z-score detection found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                detector_name=self.name,
                threshold=sensitivity,
                metadata={
                    "window": self.window,
                    "mean": mean_val,
                    "std": std_val,
                    "baseline_used": use_baseline,
                    "memory_applied": memory is not None,
                }
            )

        except Exception as e:
            logger.error(f"Z-score detection failed: {e}")
            return DetectionResult(success=False, error=str(e))
