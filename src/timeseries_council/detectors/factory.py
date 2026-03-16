# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Factory for creating detector instances.
"""

from typing import Dict, Type, Optional, List, Any
from .base import BaseDetector
from ..logging import get_logger
from ..exceptions import DetectorError

logger = get_logger(__name__)


# Registry of available detectors
_DETECTORS: Dict[str, Type[BaseDetector]] = {}


def _get_detectors() -> Dict[str, Type[BaseDetector]]:
    """Lazily load detector classes."""
    global _DETECTORS
    if not _DETECTORS:
        from .zscore import ZScoreDetector
        from .mad import MADDetector
        from .isolation_forest import IsolationForestDetector
        from .lof import LOFDetector
        from .merlion_detectors import (
            MerlionDetector, WindStatsDetector,
            SpectralResidualDetector, ProphetDetector
        )
        from .lstm_vae import LSTMVAEDetector
        from .llm_detector import LLMDetector
        from .rule_detector import RuleDetector
        from .moirai import MoiraiAnomalyDetector
        from .pyod_detectors import (
            ECODDetector, COPODDetector, HBOSDetector,
            KNNDetector, OCSVMDetector, LODADetector
        )

        _DETECTORS = {
            "zscore": ZScoreDetector,
            "z-score": ZScoreDetector,  # alias
            "mad": MADDetector,
            "isolation-forest": IsolationForestDetector,
            "isolationforest": IsolationForestDetector,  # alias
            "lof": LOFDetector,
            "merlion": MerlionDetector,
            "windstats": WindStatsDetector,
            "spectral": SpectralResidualDetector,
            "prophet": ProphetDetector,
            "lstm-vae": LSTMVAEDetector,
            "lstmvae": LSTMVAEDetector,  # alias
            "llm": LLMDetector,
            "moirai": MoiraiAnomalyDetector,
            "moirai-anomaly": MoiraiAnomalyDetector,  # alias
            "ecod": ECODDetector,
            "copod": COPODDetector,
            "hbos": HBOSDetector,
            "knn": KNNDetector,
            "ocsvm": OCSVMDetector,
            "one-class-svm": OCSVMDetector,  # alias
            "loda": LODADetector,
            "rule-detector": RuleDetector,
            "rule": RuleDetector,  # alias
        }
    return _DETECTORS


def create_detector(
    detector_name: str,
    auto_setup: bool = True,
    **kwargs: Any
) -> BaseDetector:
    """
    Factory function to create detector instances.

    Args:
        detector_name: Name of the detector
            Options: 'zscore', 'mad', 'isolation-forest', 'lof',
                     'merlion', 'windstats', 'spectral', 'prophet',
                     'lstm-vae', 'llm'
        auto_setup: Auto-install missing packages (default True)
        **kwargs: Detector-specific arguments
            - zscore/mad: window (optional)
            - isolation-forest: n_estimators, contamination
            - lof: n_neighbors, contamination
            - merlion: detector_type
            - lstm-vae: window_size, latent_dim, epochs, device
            - llm: provider (required BaseLLMProvider instance)

    Returns:
        Configured detector instance

    Raises:
        DetectorError: If detector_name is not recognized
    """
    detectors = _get_detectors()
    name = detector_name.lower().strip().replace("_", "-")

    if name not in detectors:
        available = get_available_detectors()
        logger.warning(f"Unknown detector: {detector_name}")
        return None

    # Auto-setup if enabled
    if auto_setup:
        try:
            from ..setup_models import ensure_packages_installed
            setup_result = ensure_packages_installed(name, auto_install=True)
            if not setup_result["success"]:
                logger.warning(f"Auto-setup for {name} failed: {setup_result['message']}")
        except Exception as e:
            logger.warning(f"Auto-setup failed: {e}")

    detector_class = detectors[name]
    logger.info(f"Creating detector: {detector_name}")

    return detector_class(**kwargs)


def get_available_detectors() -> List[str]:
    """Return list of available detector names (excluding aliases)."""
    detectors = _get_detectors()
    aliases = {"z-score", "isolationforest", "lstmvae", "moirai-anomaly", "one-class-svm"}
    return [k for k in detectors.keys() if k not in aliases]


# Alias for backwards compatibility
list_detectors = get_available_detectors


def get_detector_info() -> Dict[str, Dict[str, Any]]:
    """Return information about all available detectors."""
    return {
        "zscore": {
            "name": "Z-Score",
            "description": "Statistical Z-score based detection",
            "requires": [],
            "complexity": "O(n)",
            "best_for": "Simple point anomalies with normal distribution"
        },
        "mad": {
            "name": "MAD",
            "description": "Median Absolute Deviation (robust to outliers)",
            "requires": [],
            "complexity": "O(n)",
            "best_for": "Data with existing outliers, non-normal distribution"
        },
        "isolation-forest": {
            "name": "Isolation Forest",
            "description": "Ensemble tree-based isolation",
            "requires": ["scikit-learn"],
            "complexity": "O(n log n)",
            "best_for": "High-dimensional anomalies, contextual anomalies"
        },
        "lof": {
            "name": "Local Outlier Factor",
            "description": "Density-based local outlier detection",
            "requires": ["scikit-learn"],
            "complexity": "O(n²)",
            "best_for": "Local density anomalies, varying densities"
        },
        "windstats": {
            "name": "WindStats (Merlion)",
            "description": "Window-based statistical anomaly detection",
            "requires": ["salesforce-merlion"],
            "complexity": "O(n)",
            "best_for": "Time series with regular patterns"
        },
        "spectral": {
            "name": "Spectral Residual (Merlion)",
            "description": "Frequency-domain anomaly detection",
            "requires": ["salesforce-merlion"],
            "complexity": "O(n log n)",
            "best_for": "Periodic time series, spectral anomalies"
        },
        "prophet": {
            "name": "Prophet (Merlion)",
            "description": "Forecast-based detection using Prophet",
            "requires": ["salesforce-merlion", "prophet"],
            "complexity": "O(n)",
            "best_for": "Seasonal time series, trend changes"
        },
        "lstm-vae": {
            "name": "LSTM-VAE",
            "description": "Deep learning VAE with LSTM encoder",
            "requires": ["torch"],
            "complexity": "O(n × epochs)",
            "best_for": "Complex temporal patterns, sequence anomalies"
        },
        "llm": {
            "name": "LLM Detector",
            "description": "LLM-based contextual anomaly detection",
            "requires": ["provider instance"],
            "complexity": "Varies (API calls)",
            "best_for": "Contextual analysis, explainable detection"
        },
        "moirai": {
            "name": "Moirai2 Anomaly",
            "description": "Foundation model back-prediction anomaly detection using Moirai2",
            "requires": ["uni2ts", "gluonts", "torch"],
            "complexity": "O(n × inference)",
            "best_for": "Complex temporal patterns, probabilistic anomaly scoring"
        },
        "ecod": {
            "name": "ECOD",
            "description": "Empirical CDF-based outlier detection (parameter-free)",
            "requires": ["pyod"],
            "complexity": "O(n × d)",
            "best_for": "General anomalies, parameter-free detection"
        },
        "copod": {
            "name": "COPOD",
            "description": "Copula-based outlier detection (parameter-free)",
            "requires": ["pyod"],
            "complexity": "O(n × d)",
            "best_for": "Multivariate anomalies, parameter-free detection"
        },
        "hbos": {
            "name": "HBOS",
            "description": "Histogram-based outlier score (very fast)",
            "requires": ["pyod"],
            "complexity": "O(n × d)",
            "best_for": "Fast baseline, univariate anomalies"
        },
        "knn": {
            "name": "KNN",
            "description": "k-Nearest Neighbors outlier detection",
            "requires": ["pyod"],
            "complexity": "O(n²)",
            "best_for": "Distance-based anomalies, complementary to LOF"
        },
        "ocsvm": {
            "name": "OCSVM",
            "description": "One-Class SVM outlier detection",
            "requires": ["pyod"],
            "complexity": "O(n² ~ n³)",
            "best_for": "High-dimensional feature spaces, kernel-based detection"
        },
        "loda": {
            "name": "LODA",
            "description": "Lightweight online detector using random projections",
            "requires": ["pyod"],
            "complexity": "O(n × d × k)",
            "best_for": "Fast ensemble detection, streaming scenarios"
        },
    }
