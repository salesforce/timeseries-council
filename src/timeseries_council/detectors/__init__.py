# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Anomaly detection models for time series data.

Available detectors:
- ZScoreDetector: Statistical Z-score based detection
- MADDetector: Median Absolute Deviation (robust)
- IsolationForestDetector: Ensemble tree-based isolation
- LOFDetector: Local Outlier Factor (density-based)
- MerlionDetector: Merlion library wrappers
- LSTMVAEDetector: LSTM Variational Autoencoder
- LLMDetector: LLM-based contextual detection
- EnsembleDetector: Combine multiple detectors
- ECODDetector: Empirical CDF-based (PyOD)
- COPODDetector: Copula-based (PyOD)
- HBOSDetector: Histogram-based (PyOD)
- KNNDetector: k-Nearest Neighbors (PyOD)
- OCSVMDetector: One-Class SVM (PyOD)
- LODADetector: Lightweight Online Detector (PyOD)
"""

from .base import BaseDetector, EnsembleDetector
from .factory import create_detector, get_available_detectors, get_detector_info, list_detectors

__all__ = [
    "BaseDetector",
    "EnsembleDetector",
    "create_detector",
    "get_available_detectors",
    "get_detector_info",
    "list_detectors",
]
