# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Multi-Model Orchestration for timeseries-council.

Smart model selection and ensemble execution for forecasting and detection.
"""

from .characteristics import CharacteristicsAnalyzer, DataCharacteristics
from .selector import ModelSelector, ModelSelection
from .ensemble import MultiModelEnsemble, EnsembleResult
from .cross_validation import (
    CrossValResult,
    CVSelectionResult,
    run_cross_validation,
    select_model_with_cv,
    check_prediction_diversity,
)
from .quantile_ensemble import (
    QuantileEnsembleResult,
    quantile_ensemble,
    collect_quantile_predictions,
    mixture_blend,
)

__all__ = [
    # Characteristics
    "CharacteristicsAnalyzer",
    "DataCharacteristics",
    # Selector
    "ModelSelector",
    "ModelSelection",
    # Ensemble
    "MultiModelEnsemble",
    "EnsembleResult",
    # Cross-validation
    "CrossValResult",
    "CVSelectionResult",
    "run_cross_validation",
    "select_model_with_cv",
    "check_prediction_diversity",
    # Quantile ensemble
    "QuantileEnsembleResult",
    "quantile_ensemble",
    "collect_quantile_predictions",
    "mixture_blend",
]
