# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Local Outlier Factor (LOF) anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)


class LOFDetector(BaseDetector):
    """Local Outlier Factor based anomaly detector.

    Measures the local density deviation of a data point
    with respect to its neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: str = "auto"
    ):
        """
        Initialize LOF detector.

        Args:
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of anomalies ('auto' or float)
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        logger.info(f"Initialized LOFDetector (n_neighbors={n_neighbors})")

    @property
    def name(self) -> str:
        return "LOF"

    @property
    def description(self) -> str:
        return "Local Outlier Factor detector measuring local density deviation"

    def _prepare_features(
        self,
        series: pd.Series,
        memory: Optional[DetectionMemory] = None,
    ) -> np.ndarray:
        """Prepare feature matrix for LOF.

        When memory provides baseline_stats (mean/std), an extra
        ``baseline_zscore`` feature is appended so the model can
        learn that points far from the known-normal baseline are
        more likely anomalous.
        """
        df = pd.DataFrame({'value': series})

        # Add lag features
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = series.shift(lag)

        # Add differences
        df['diff_1'] = series.diff(1)

        # Add rolling mean
        df['rolling_mean_3'] = series.rolling(3).mean()

        # Baseline z-score feature (deep memory integration)
        if (
            memory is not None
            and memory.baseline_stats.get("mean") is not None
            and memory.baseline_stats.get("std") is not None
            and memory.baseline_stats["std"] > 0
        ):
            b_mean = memory.baseline_stats["mean"]
            b_std = memory.baseline_stats["std"]
            df['baseline_zscore'] = (series - b_mean) / b_std
            logger.info("Added baseline_zscore feature to LOF")

        # Drop NaN rows
        df = df.dropna()
        return df.values, df.index

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using Local Outlier Factor."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading sklearn...", 0.1)

        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            logger.error("scikit-learn not installed")
            return DetectionResult(
                success=False,
                error="scikit-learn not installed. Run: pip install scikit-learn"
            )

        try:
            self._report_progress(progress_callback, "Preparing features...", 0.2)

            # Prepare features (with optional baseline_zscore from memory)
            X, valid_index = self._prepare_features(series, memory)

            # Adjust n_neighbors if needed
            n_neighbors = min(self.n_neighbors, len(X) - 1)
            if n_neighbors < 2:
                return DetectionResult(
                    success=False,
                    error="Need at least 3 valid data points after feature engineering"
                )

            self._report_progress(progress_callback, "Computing LOF scores...", 0.4)

            # Convert sensitivity to contamination
            if self.contamination == "auto":
                contamination = min(0.5, max(0.01, 1.0 / (sensitivity * 2)))
            else:
                contamination = self.contamination

            # Fit model
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                n_jobs=-1
            )

            predictions = model.fit_predict(X)
            scores = -model.negative_outlier_factor_  # Higher = more anomalous

            self._report_progress(progress_callback, "Processing results...", 0.8)

            # Extract anomalies
            anomalies = []
            mean_val = float(series.mean())

            for i, (idx, pred) in enumerate(zip(valid_index, predictions)):
                if pred == -1:  # Anomaly
                    val = float(series[idx])
                    score = float(scores[i])
                    anomaly_type = AnomalyType.SPIKE if val > mean_val else AnomalyType.DROP

                    anomalies.append(Anomaly(
                        timestamp=str(idx),
                        value=val,
                        score=score,
                        anomaly_type=anomaly_type
                    ))

            # Apply memory context
            anomalies = self._apply_memory(anomalies, memory)

            self._report_progress(progress_callback, "Detection complete", 1.0)

            logger.info(f"LOF found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
                    "n_neighbors": n_neighbors,
                    "contamination": contamination,
                    "mean": mean_val,
                    "std": float(series.std()),
                    "baseline_used": (
                        memory is not None
                        and bool(memory.baseline_stats.get("mean") is not None)
                    ),
                    "memory_applied": memory is not None,
                }
            )

        except Exception as e:
            logger.error(f"LOF detection failed: {e}")
            return DetectionResult(success=False, error=str(e))
