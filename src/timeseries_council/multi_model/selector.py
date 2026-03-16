# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Model Selector - Smart selection of models based on data characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from .characteristics import CharacteristicsAnalyzer, DataCharacteristics
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSelection:
    """Result of model selection."""
    models: List[str] = field(default_factory=list)
    reasoning: str = ""
    characteristics: Optional[DataCharacteristics] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models": self.models,
            "reasoning": self.reasoning,
            "characteristics": self.characteristics.to_dict() if self.characteristics else None,
            "confidence": self.confidence,
        }


class ModelSelector:
    """
    Select appropriate models based on data characteristics.

    Can use:
    1. Rule-based heuristics (fast, no LLM needed)
    2. LLM-based selection (more sophisticated)
    """

    # Available models by category
    FORECASTERS = {
        "moirai": {
            "strengths": ["short series", "general purpose", "fast"],
            "weaknesses": ["very long series"],
            "min_length": 10,
            "max_length": 5000,
        },
        "chronos2": {
            "strengths": ["seasonality", "general purpose", "uncertainty"],
            "weaknesses": ["very short series"],
            "min_length": 20,
            "max_length": 10000,
        },
        "timesfm": {
            "strengths": ["long series", "trend", "fast"],
            "weaknesses": ["short series"],
            "min_length": 50,
            "max_length": 100000,
        },
        "lag_llama": {
            "strengths": ["complex patterns", "volatility"],
            "weaknesses": ["slow", "short series"],
            "min_length": 100,
            "max_length": 10000,
        },
        "tirex": {
            "strengths": ["quantile forecasting", "general purpose", "uncertainty"],
            "weaknesses": ["very short series"],
            "min_length": 20,
            "max_length": 10000,
        },
        "zscore_baseline": {
            "strengths": ["fast", "interpretable", "any length"],
            "weaknesses": ["simple patterns only"],
            "min_length": 5,
            "max_length": 1000000,
        },
    }

    DETECTORS = {
        "zscore": {
            "strengths": ["fast", "normal distribution", "interpretable"],
            "weaknesses": ["non-normal data", "contextual anomalies"],
        },
        "mad": {
            "strengths": ["robust to outliers", "non-normal data"],
            "weaknesses": ["may miss subtle anomalies"],
        },
        "isolation_forest": {
            "strengths": ["complex patterns", "high dimensional"],
            "weaknesses": ["slower", "less interpretable"],
        },
        "lof": {
            "strengths": ["local anomalies", "density-based"],
            "weaknesses": ["slow for large data", "parameter sensitive"],
        },
    }

    LLM_SELECTION_PROMPT = """You are a time series model selection expert.

Given the following data characteristics, select 3-5 appropriate models.

DATA CHARACTERISTICS:
{characteristics}

AVAILABLE FORECASTING MODELS:
{forecasters}

AVAILABLE DETECTION MODELS:
{detectors}

TASK TYPE: {task_type}

Select the best models for this data and explain your reasoning.

Respond in JSON format:
```json
{{
    "models": ["model1", "model2", "model3"],
    "reasoning": "Brief explanation of why these models were selected"
}}
```"""

    def __init__(self, llm_provider: Optional[Any] = None):
        """
        Initialize the model selector.

        Args:
            llm_provider: Optional LLM for sophisticated selection
        """
        self.llm_provider = llm_provider
        self.analyzer = CharacteristicsAnalyzer()

    def select_forecasters(
        self,
        characteristics: DataCharacteristics,
        count: int = 4,
        use_llm: bool = False,
    ) -> ModelSelection:
        """
        Select forecasting models based on data characteristics.

        Args:
            characteristics: Analyzed data characteristics
            count: Number of models to select (3-5)
            use_llm: Whether to use LLM for selection

        Returns:
            ModelSelection with selected models
        """
        count = max(3, min(5, count))

        if use_llm and self.llm_provider:
            return self._llm_select(characteristics, "forecasting", count)

        return self._rule_based_select_forecasters(characteristics, count)

    def select_detectors(
        self,
        characteristics: DataCharacteristics,
        count: int = 3,
        use_llm: bool = False,
    ) -> ModelSelection:
        """
        Select anomaly detection models based on data characteristics.

        Args:
            characteristics: Analyzed data characteristics
            count: Number of models to select (3-5)
            use_llm: Whether to use LLM for selection

        Returns:
            ModelSelection with selected models
        """
        count = max(2, min(4, count))

        if use_llm and self.llm_provider:
            return self._llm_select(characteristics, "detection", count)

        return self._rule_based_select_detectors(characteristics, count)

    def _rule_based_select_forecasters(
        self,
        characteristics: DataCharacteristics,
        count: int,
    ) -> ModelSelection:
        """Rule-based forecaster selection."""
        scores: Dict[str, float] = {}

        for model, props in self.FORECASTERS.items():
            score = 0.0

            # Length compatibility
            if props["min_length"] <= characteristics.length <= props["max_length"]:
                score += 2.0
            elif characteristics.length < props["min_length"]:
                score -= 1.0

            # Strength matching
            if characteristics.length < 100 and "short series" in props["strengths"]:
                score += 1.5
            if characteristics.length > 1000 and "long series" in props["strengths"]:
                score += 1.5
            if characteristics.has_seasonality and "seasonality" in props["strengths"]:
                score += 1.0
            if characteristics.has_trend and "trend" in props["strengths"]:
                score += 1.0
            if characteristics.volatility_level == "high" and "volatility" in props["strengths"]:
                score += 1.0

            # Speed preference for interactive use
            if "fast" in props["strengths"]:
                score += 0.5

            scores[model] = score

        # Sort by score and select top N
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [m for m, s in sorted_models[:count]]

        # Generate reasoning
        reasoning_parts = []
        if characteristics.length < 100:
            reasoning_parts.append("short series favors simpler models")
        if characteristics.has_seasonality:
            reasoning_parts.append("seasonality detected, included models with seasonal handling")
        if characteristics.volatility_level == "high":
            reasoning_parts.append("high volatility suggests uncertainty-aware models")

        reasoning = f"Selected based on: {', '.join(reasoning_parts)}" if reasoning_parts else "General selection based on data characteristics"

        return ModelSelection(
            models=selected,
            reasoning=reasoning,
            characteristics=characteristics,
            confidence=0.8,
        )

    def _rule_based_select_detectors(
        self,
        characteristics: DataCharacteristics,
        count: int,
    ) -> ModelSelection:
        """Rule-based detector selection."""
        scores: Dict[str, float] = {}

        for model, props in self.DETECTORS.items():
            score = 1.0  # Base score

            # Normal distribution check
            if abs(characteristics.skewness) < 0.5 and abs(characteristics.kurtosis) < 1:
                if "normal distribution" in props["strengths"]:
                    score += 1.0
            else:
                if "non-normal data" in props["strengths"]:
                    score += 1.0

            # Volatility
            if characteristics.volatility_level == "high":
                if "robust to outliers" in props["strengths"]:
                    score += 1.0

            # Length considerations
            if characteristics.length > 10000:
                if "slow" in props["weaknesses"]:
                    score -= 0.5

            # Speed
            if "fast" in props["strengths"]:
                score += 0.3

            scores[model] = score

        # Sort and select
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [m for m, s in sorted_models[:count]]

        reasoning = "Selected detectors based on data distribution and length"

        return ModelSelection(
            models=selected,
            reasoning=reasoning,
            characteristics=characteristics,
            confidence=0.8,
        )

    def _llm_select(
        self,
        characteristics: DataCharacteristics,
        task_type: str,
        count: int,
    ) -> ModelSelection:
        """Use LLM for model selection."""
        prompt = self.LLM_SELECTION_PROMPT.format(
            characteristics=characteristics.summary(),
            forecasters=json.dumps(self.FORECASTERS, indent=2),
            detectors=json.dumps(self.DETECTORS, indent=2),
            task_type=task_type,
        )

        try:
            response = self.llm_provider.generate(prompt)

            # Parse response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return ModelSelection(
                    models=data.get("models", [])[:count],
                    reasoning=data.get("reasoning", "LLM-based selection"),
                    characteristics=characteristics,
                    confidence=0.9,
                )
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}, falling back to rules")

        # Fallback to rule-based
        if task_type == "forecasting":
            return self._rule_based_select_forecasters(characteristics, count)
        else:
            return self._rule_based_select_detectors(characteristics, count)
