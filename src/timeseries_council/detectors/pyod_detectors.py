# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
PyOD library based anomaly detectors.

Implements six detectors from the PyOD library (Python Outlier Detection),
selected based on ADBench (NeurIPS 2022) top performers and algorithm diversity:
- ECOD: Empirical CDF-based outlier detection (parameter-free)
- COPOD: Copula-based outlier detection (parameter-free)
- HBOS: Histogram-based outlier score (very fast)
- KNN: k-Nearest Neighbors outlier detection
- OCSVM: One-Class SVM
- LODA: Lightweight Online Detector of Anomalies
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger

logger = get_logger(__name__)


class BasePyODDetector(BaseDetector):
    """Base class for all PyOD-based anomaly detectors.

    Handles shared logic: feature engineering, sensitivity-to-contamination
    conversion, model fitting, and result extraction. Subclasses only need
    to implement _create_model() and the name/description properties.
    """

    def _prepare_features(
        self,
        series: pd.Series,
        memory: Optional[DetectionMemory] = None,
    ):
        """Prepare feature matrix from 1D time series.

        Creates: value, lag_1..3, rolling_mean_3/7, rolling_std_3/7, diff_1, diff_2.
        When memory provides baseline_stats (mean/std), an extra
        ``baseline_zscore`` feature is appended so the model can learn
        that points far from the known-normal baseline are more anomalous.
        """
        df = pd.DataFrame({'value': series})

        for lag in [1, 2, 3]:
            df[f'lag_{lag}'] = series.shift(lag)

        for window in [3, 7]:
            df[f'rolling_mean_{window}'] = series.rolling(window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window).std()

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
            logger.info(f"Added baseline_zscore feature to {self.name}")

        df = df.dropna()
        return df.values, df.index

    def _sensitivity_to_contamination(self, sensitivity: float) -> float:
        """Convert sensitivity parameter to PyOD contamination ratio.

        Same formula as IsolationForestDetector and LOFDetector.
        """
        return min(0.5, max(0.01, 1.0 / (sensitivity * 2)))

    def _create_model(self, contamination: float):
        """Create a PyOD model instance. Must be implemented by subclasses."""
        raise NotImplementedError

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using a PyOD model."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Loading pyod...", 0.1)

        try:
            import pyod  # noqa: F401
        except ImportError:
            logger.error("pyod not installed")
            return DetectionResult(
                success=False,
                error="pyod not installed. Run: pip install pyod"
            )

        try:
            self._report_progress(progress_callback, "Preparing features...", 0.2)

            X, valid_index = self._prepare_features(series, memory)

            if len(X) < 10:
                return DetectionResult(
                    success=False,
                    error="Need at least 10 valid data points after feature engineering"
                )

            self._report_progress(progress_callback, f"Training {self.name}...", 0.4)

            contamination = self._sensitivity_to_contamination(sensitivity)

            model = self._create_model(contamination)
            model.fit(X)

            labels = model.labels_
            scores = model.decision_scores_

            self._report_progress(progress_callback, "Processing results...", 0.8)

            anomalies = []
            mean_val = float(series.mean())

            for i, (idx, label) in enumerate(zip(valid_index, labels)):
                if label == 1:
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

            logger.info(f"{self.name} found {len(anomalies)} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
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
            logger.error(f"{self.name} detection failed: {e}")
            return DetectionResult(success=False, error=str(e))


class ECODDetector(BasePyODDetector):
    """ECOD (Empirical Cumulative Distribution) based anomaly detector.

    Parameter-free, fast, and a top performer in ADBench (NeurIPS 2022).
    """

    @property
    def name(self) -> str:
        return "ECOD"

    @property
    def description(self) -> str:
        return "Empirical CDF-based outlier detection (parameter-free)"

    def _create_model(self, contamination: float):
        from pyod.models.ecod import ECOD
        return ECOD(contamination=contamination)


class COPODDetector(BasePyODDetector):
    """COPOD (Copula-Based Outlier Detection) anomaly detector.

    Parameter-free, fast, strong on multivariate data.
    """

    @property
    def name(self) -> str:
        return "COPOD"

    @property
    def description(self) -> str:
        return "Copula-based outlier detection (parameter-free)"

    def _create_model(self, contamination: float):
        from pyod.models.copod import COPOD
        return COPOD(contamination=contamination)


class HBOSDetector(BasePyODDetector):
    """HBOS (Histogram-Based Outlier Score) anomaly detector.

    Very fast histogram-based approach.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        logger.info(f"Initialized HBOSDetector (n_bins={n_bins})")

    @property
    def name(self) -> str:
        return "HBOS"

    @property
    def description(self) -> str:
        return "Histogram-based outlier score detection"

    def _create_model(self, contamination: float):
        from pyod.models.hbos import HBOS
        return HBOS(n_bins=self.n_bins, contamination=contamination)


class KNNDetector(BasePyODDetector):
    """KNN (k-Nearest Neighbors) anomaly detector.

    Classic proximity-based method, complementary to LOF.
    Uses distance to k-th nearest neighbor as anomaly score.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        logger.info(f"Initialized KNNDetector (n_neighbors={n_neighbors})")

    @property
    def name(self) -> str:
        return "KNN"

    @property
    def description(self) -> str:
        return "k-Nearest Neighbors outlier detection"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect with dynamic n_neighbors adjustment."""
        # Pre-check: adjust n_neighbors to data size (same pattern as LOFDetector)
        X_check, _ = self._prepare_features(series, memory)
        if len(X_check) > 0:
            self._effective_neighbors = min(self.n_neighbors, len(X_check) - 1)
        else:
            self._effective_neighbors = self.n_neighbors
        return super().detect(series, sensitivity, progress_callback, memory=memory)

    def _create_model(self, contamination: float):
        from pyod.models.knn import KNN
        return KNN(
            n_neighbors=self._effective_neighbors,
            contamination=contamination
        )


class OCSVMDetector(BasePyODDetector):
    """OCSVM (One-Class SVM) anomaly detector.

    Learns a decision boundary around normal data.
    """

    def __init__(self, kernel: str = "rbf"):
        self.kernel = kernel
        logger.info(f"Initialized OCSVMDetector (kernel={kernel})")

    @property
    def name(self) -> str:
        return "OCSVM"

    @property
    def description(self) -> str:
        return "One-Class SVM outlier detection"

    def _create_model(self, contamination: float):
        from pyod.models.ocsvm import OCSVM
        return OCSVM(kernel=self.kernel, contamination=contamination)


class LODADetector(BasePyODDetector):
    """LODA (Lightweight Online Detector of Anomalies) anomaly detector.

    Ensemble of random projection histograms. Fast and robust.
    """

    @property
    def name(self) -> str:
        return "LODA"

    @property
    def description(self) -> str:
        return "Lightweight online detector using random projection histograms"

    def _create_model(self, contamination: float):
        from pyod.models.loda import LODA
        return LODA(contamination=contamination)
