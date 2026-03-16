# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Merlion library based anomaly detectors.
"""

import sys
import os
from typing import Optional, Callable
import pandas as pd

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)

# Add custom Merlion path if set via environment variable
MERLION_PATH = os.environ.get("MERLION_PATH", "")
if MERLION_PATH and os.path.exists(MERLION_PATH) and MERLION_PATH not in sys.path:
    sys.path.insert(0, MERLION_PATH)


class MerlionDetector(BaseDetector):
    """Base wrapper for Merlion anomaly detectors."""

    def __init__(self, detector_type: str = "default"):
        """
        Initialize Merlion detector.

        Args:
            detector_type: Type of Merlion detector to use
                Options: 'default', 'prophet', 'windstats', 'spectral'
        """
        self.detector_type = detector_type
        self._model = None
        logger.info(f"Initialized MerlionDetector: {detector_type}")

    @property
    def name(self) -> str:
        return f"Merlion-{self.detector_type}"

    @property
    def description(self) -> str:
        return f"Merlion {self.detector_type} anomaly detector"

    def _get_detector(self, threshold):
        """Create Merlion detector instance."""
        try:
            from merlion.models.anomaly.forecast_based.prophet import (
                ProphetDetector, ProphetDetectorConfig
            )
            from merlion.models.anomaly.windstats import (
                WindStats, WindStatsConfig
            )
            from merlion.models.anomaly.spectral_residual import (
                SpectralResidual, SpectralResidualConfig
            )
            from merlion.models.anomaly.isolation_forest import (
                IsolationForest as MerlionIF, IsolationForestConfig
            )
            from merlion.post_process.threshold import AggregateAlarms

            threshold_config = AggregateAlarms(alm_threshold=threshold)

            if self.detector_type == "prophet":
                config = ProphetDetectorConfig(threshold=threshold_config)
                return ProphetDetector(config)
            elif self.detector_type == "windstats":
                config = WindStatsConfig(threshold=threshold_config)
                return WindStats(config)
            elif self.detector_type == "spectral":
                config = SpectralResidualConfig(threshold=threshold_config)
                return SpectralResidual(config)
            elif self.detector_type == "isolation_forest":
                config = IsolationForestConfig(threshold=threshold_config)
                return MerlionIF(config)
            else:
                # Default to WindStats
                config = WindStatsConfig(threshold=threshold_config)
                return WindStats(config)

        except ImportError as e:
            raise ImportError(
                f"Merlion not properly installed: {e}. "
                "Install salesforce-merlion package."
            )

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using Merlion detector."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading Merlion...", 0.1)

        try:
            from merlion.utils import TimeSeries
        except ImportError:
            return DetectionResult(
                success=False,
                error="Merlion not installed. Install salesforce-merlion package."
            )

        try:
            self._report_progress(progress_callback, "Preparing data...", 0.2)

            # Convert sensitivity to threshold
            threshold = 3.0 / sensitivity  # Higher sensitivity = lower threshold

            # Deep memory integration: when baseline std is known, scale
            # the threshold so detection is relative to baseline variation.
            # If current data has higher variance than baseline, lower the
            # threshold to catch more deviations from normal.
            use_baseline = (
                memory is not None
                and memory.baseline_stats.get("std") is not None
                and memory.baseline_stats["std"] > 0
            )
            if use_baseline:
                current_std = float(series.std())
                baseline_std = memory.baseline_stats["std"]
                if current_std > 0:
                    ratio = baseline_std / current_std
                    threshold *= ratio
                    logger.info(
                        f"Scaled Merlion threshold by baseline/current std "
                        f"ratio ({baseline_std:.2f}/{current_std:.2f}={ratio:.2f}), "
                        f"new threshold={threshold:.4f}"
                    )

            # Create detector
            self._report_progress(progress_callback, f"Creating {self.detector_type} detector...", 0.3)
            model = self._get_detector(threshold)

            # Prepare time series - normalize timezone to avoid comparison issues
            # Merlion expects timezone-naive datetimes
            working_series = series.copy()
            original_tz = None
            if hasattr(working_series.index, 'tz') and working_series.index.tz is not None:
                original_tz = working_series.index.tz
                working_series.index = working_series.index.tz_localize(None)
                logger.info(f"Converted timezone-aware index ({original_tz}) to timezone-naive for Merlion")

            target_df = working_series.to_frame()
            train_ts = TimeSeries.from_pd(target_df)

            self._report_progress(progress_callback, "Training detector...", 0.5)

            # Train
            model.train(train_ts)

            self._report_progress(progress_callback, "Getting anomaly scores...", 0.7)

            # Get anomaly scores
            scores = model.get_anomaly_label(train_ts)

            if hasattr(scores, 'to_pd'):
                scores_df = scores.to_pd()
            else:
                scores_df = scores

            self._report_progress(progress_callback, "Processing results...", 0.9)

            # Extract anomalies (where score > 0)
            anomalies = []
            mean_val = float(working_series.mean())

            # Baseline z-score helper
            b_mean = memory.baseline_stats.get("mean") if use_baseline else None
            b_std = memory.baseline_stats.get("std") if use_baseline else None

            score_col = scores_df.columns[0]
            for idx, score_val in scores_df[score_col].items():
                if score_val > 0:
                    if idx in working_series.index:
                        val = float(working_series[idx])
                        anomaly_type = AnomalyType.SPIKE if val > mean_val else AnomalyType.DROP

                        # Boost score with baseline deviation when available
                        final_score = float(score_val)
                        if b_mean is not None and b_std is not None and b_std > 0:
                            baseline_z = abs(val - b_mean) / b_std
                            final_score = max(final_score, baseline_z)

                        # Convert timestamp back to original timezone if needed
                        if original_tz is not None:
                            timestamp_str = str(pd.Timestamp(idx).tz_localize(original_tz))
                        else:
                            timestamp_str = str(idx)

                        anomalies.append(Anomaly(
                            timestamp=timestamp_str,
                            value=val,
                            score=final_score,
                            anomaly_type=anomaly_type
                        ))

            # Apply memory context
            anomalies = self._apply_memory(anomalies, memory)

            self._report_progress(progress_callback, "Detection complete", 1.0)

            logger.info(f"Merlion {self.detector_type} found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
                    "detector_type": self.detector_type,
                    "mean": mean_val,
                    "std": float(working_series.std()),
                    "timezone_normalized": original_tz is not None,
                    "baseline_used": use_baseline,
                    "memory_applied": memory is not None,
                }
            )

        except Exception as e:
            logger.error(f"Merlion detection failed: {e}")
            return DetectionResult(success=False, error=str(e))


class WindStatsDetector(MerlionDetector):
    """Merlion WindStats detector."""

    def __init__(self):
        super().__init__("windstats")


class SpectralResidualDetector(MerlionDetector):
    """Merlion Spectral Residual detector."""

    def __init__(self):
        super().__init__("spectral")


class ProphetDetector(MerlionDetector):
    """Merlion Prophet-based detector."""

    def __init__(self):
        super().__init__("prophet")
