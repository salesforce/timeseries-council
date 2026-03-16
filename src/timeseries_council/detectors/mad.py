# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Median Absolute Deviation (MAD) based anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)


class MADDetector(BaseDetector):
    """Median Absolute Deviation based anomaly detector.

    More robust to outliers than Z-score since it uses median
    instead of mean.
    """

    # Scale factor to make MAD comparable to standard deviation
    # for normally distributed data
    SCALE_FACTOR = 1.4826

    def __init__(self, window: Optional[int] = None):
        """
        Initialize MAD detector.

        Args:
            window: Optional rolling window size for local statistics.
                    If None, uses global statistics.
        """
        self.window = window
        logger.info(f"Initialized MADDetector (window={window})")

    @property
    def name(self) -> str:
        if self.window:
            return f"MAD-{self.window}"
        return "MAD"

    @property
    def description(self) -> str:
        return "Median Absolute Deviation detector (robust to outliers)"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 3.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using MAD method."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Calculating MAD statistics...", 0.2)

        try:
            # Deep integration: use baseline_stats from memory when available.
            # When baseline provides median and mad (or std), score each point
            # against the baseline distribution rather than the current batch.
            use_baseline = (
                memory is not None
                and memory.baseline_stats.get("median") is not None
                and (
                    memory.baseline_stats.get("mad") is not None
                    or memory.baseline_stats.get("std") is not None
                )
            )

            if use_baseline:
                baseline_median = memory.baseline_stats["median"]
                # Prefer explicit MAD; fall back to std-derived MAD
                baseline_mad = memory.baseline_stats.get("mad")
                if baseline_mad is None:
                    baseline_mad = memory.baseline_stats["std"] / self.SCALE_FACTOR
                baseline_mad_scaled = baseline_mad * self.SCALE_FACTOR
                if baseline_mad_scaled == 0:
                    baseline_mad_scaled = 1e-10
                logger.info(
                    f"Using baseline stats for MAD: median={baseline_median}, "
                    f"mad_scaled={baseline_mad_scaled}"
                )
                modified_z = (series - baseline_median) / baseline_mad_scaled
                median = baseline_median
                mad = baseline_mad_scaled
            elif self.window:
                # Rolling MAD
                rolling_median = series.rolling(window=self.window, center=True).median()

                def rolling_mad(x):
                    return np.median(np.abs(x - np.median(x)))

                rolling_mad_vals = series.rolling(window=self.window, center=True).apply(
                    rolling_mad, raw=True
                )

                # Fill edges with global stats
                global_median = series.median()
                global_mad = np.median(np.abs(series - global_median))
                rolling_median = rolling_median.fillna(global_median)
                rolling_mad_vals = rolling_mad_vals.fillna(global_mad)

                median = rolling_median
                mad = rolling_mad_vals * self.SCALE_FACTOR

                # Modified Z-score (rolling)
                mad_safe = mad.replace(0, 1e-10)
                modified_z = (series - median) / mad_safe
            else:
                # Global statistics
                median = series.median()
                mad = np.median(np.abs(series - median)) * self.SCALE_FACTOR
                if mad == 0:
                    mad = 1e-10
                modified_z = (series - median) / mad

            self._report_progress(progress_callback, "Finding anomalies...", 0.7)

            # Find anomalies
            anomaly_mask = np.abs(modified_z) > sensitivity
            anomaly_indices = series[anomaly_mask].index

            anomalies = []
            for idx in anomaly_indices:
                val = float(series[idx])
                z = float(modified_z[idx]) if isinstance(modified_z, pd.Series) else float(modified_z)
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
            median_val = float(series.median()) if not isinstance(median, pd.Series) else float(median.mean())

            logger.info(f"MAD detection found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
                    "window": self.window,
                    "median": median_val,
                    "mean": mean_val,
                    "std": std_val,
                    "scale_factor": self.SCALE_FACTOR,
                    "baseline_used": use_baseline,
                    "memory_applied": memory is not None,
                }
            )

        except Exception as e:
            logger.error(f"MAD detection failed: {e}")
            return DetectionResult(success=False, error=str(e))
