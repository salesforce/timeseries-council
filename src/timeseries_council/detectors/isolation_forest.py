# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Isolation Forest anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger
from ..exceptions import DetectorError

logger = get_logger(__name__)


class IsolationForestDetector(BaseDetector):
    """Isolation Forest based anomaly detector.

    Uses ensemble of isolation trees to detect anomalies.
    Anomalies are isolated quickly (shorter path lengths).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: str = "auto",
        random_state: Optional[int] = 42
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of anomalies ('auto' or float)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model = None
        logger.info(f"Initialized IsolationForestDetector (n_estimators={n_estimators})")

    @property
    def name(self) -> str:
        return "IsolationForest"

    @property
    def description(self) -> str:
        return "Isolation Forest detector using ensemble of isolation trees"

    def _prepare_features(
        self,
        series: pd.Series,
        memory: Optional[DetectionMemory] = None,
    ) -> np.ndarray:
        """Prepare feature matrix for Isolation Forest.

        When memory provides baseline_stats (mean/std), an extra
        ``baseline_zscore`` feature is appended so the model can
        learn that points far from the known-normal baseline are
        more likely anomalous.
        """
        # Create features: value, rolling stats, differences
        df = pd.DataFrame({'value': series})

        # Add lag features
        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = series.shift(lag)

        # Add rolling statistics
        for window in [3, 7]:
            df[f'rolling_mean_{window}'] = series.rolling(window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window).std()

        # Add differences
        df['diff_1'] = series.diff(1)
        df['diff_2'] = series.diff(2)

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
            logger.info("Added baseline_zscore feature to Isolation Forest")

        # Drop NaN rows and return
        df = df.dropna()
        return df.values, df.index

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using Isolation Forest."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading sklearn...", 0.1)

        try:
            from sklearn.ensemble import IsolationForest
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

            if len(X) < 10:
                return DetectionResult(
                    success=False,
                    error="Need at least 10 valid data points after feature engineering"
                )

            self._report_progress(progress_callback, "Training Isolation Forest...", 0.4)

            # Convert sensitivity to contamination
            # Higher sensitivity = expect fewer anomalies
            # sensitivity of 2.0 roughly corresponds to ~5% contamination
            if self.contamination == "auto":
                contamination = min(0.5, max(0.01, 1.0 / (sensitivity * 2)))
            else:
                contamination = self.contamination

            # Fit model
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=contamination,
                random_state=self.random_state,
                n_jobs=-1
            )

            predictions = model.fit_predict(X)
            scores = -model.score_samples(X)  # Higher = more anomalous

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

            logger.info(f"Isolation Forest found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
                    "n_estimators": self.n_estimators,
                    "contamination": contamination,
                    "features_used": X.shape[1],
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
            logger.error(f"Isolation Forest detection failed: {e}")
            return DetectionResult(success=False, error=str(e))
