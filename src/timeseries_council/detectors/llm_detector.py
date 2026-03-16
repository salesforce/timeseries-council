# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
LLM-based anomaly detector.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np
import json
import re

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..providers.base import BaseLLMProvider
from ..logging import get_logger

logger = get_logger(__name__)


class LLMDetector(BaseDetector):
    """Anomaly detector using Large Language Models."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        include_context: bool = True
    ):
        """
        Initialize LLM detector.

        Args:
            provider: LLM provider to use (Claude, GPT, Gemini, etc.)
            include_context: Whether to include data statistics in prompt
        """
        self.provider = provider
        self.include_context = include_context
        logger.info(f"Initialized LLMDetector with {provider.provider_name}")

    @property
    def name(self) -> str:
        return f"LLM-{self.provider.provider_name}"

    @property
    def description(self) -> str:
        return f"LLM-based anomaly detector using {self.provider.provider_name}"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies using LLM analysis."""
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)

        self._report_progress(progress_callback, "Preparing analysis...", 0.1)

        try:
            # Calculate statistics
            mean_val = float(series.mean())
            std_val = float(series.std())

            # Pre-filter potential anomalies with Z-score
            z_scores = (series - mean_val) / std_val
            potential_anomalies = series[np.abs(z_scores) > 1.5]  # Broader initial filter

            if len(potential_anomalies) == 0:
                self._report_progress(progress_callback, "No potential anomalies found", 1.0)
                return DetectionResult(
                    success=True,
                    anomaly_count=0,
                    anomalies=[],
                    threshold=sensitivity,
                    detector_name=self.name,
                    metadata={"llm_analyzed": False, "mean": mean_val, "std": std_val}
                )

            self._report_progress(progress_callback, "Analyzing with LLM...", 0.3)

            # Build analysis prompt
            prompt = self._build_prompt(series, potential_anomalies, sensitivity, memory)

            system_instruction = """You are an expert time series analyst specializing in anomaly detection.
Analyze the provided data and identify genuine anomalies.
Consider: point anomalies, contextual anomalies, and pattern breaks.

Return your response as a valid JSON object with this structure:
{
    "anomalies": [
        {
            "timestamp": "timestamp string",
            "value": numeric_value,
            "type": "spike" or "drop" or "shift",
            "confidence": 0.0-1.0,
            "reason": "brief explanation"
        }
    ],
    "summary": "brief analysis summary"
}

IMPORTANT: Return ONLY the JSON object. Timestamps must exactly match the input format."""

            # Call LLM
            response = self.provider.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.2
            )

            self._report_progress(progress_callback, "Processing results...", 0.7)

            # Parse response
            result = self._parse_response(response, sensitivity)

            if result is None:
                # Fallback to basic Z-score detection
                logger.warning("Failed to parse LLM response, using Z-score fallback")
                return self._fallback_detection(series, sensitivity, mean_val, std_val)

            # Apply memory context
            result["anomalies"] = self._apply_memory(result["anomalies"], memory)

            self._report_progress(progress_callback, "Detection complete", 1.0)

            logger.info(f"LLM detector found {len(result['anomalies'])} anomalies")

            return DetectionResult(
                success=True,
                anomaly_count=len(result["anomalies"]),
                anomalies=result["anomalies"],
                threshold=sensitivity,
                detector_name=self.name,
                metadata={
                    "llm_analyzed": True,
                    "summary": result.get("summary", ""),
                    "memory_applied": memory is not None,
                    "mean": mean_val,
                    "std": std_val,
                }
            )

        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return DetectionResult(success=False, error=str(e))

    def _build_prompt(
        self,
        series: pd.Series,
        potential_anomalies: pd.Series,
        sensitivity: float,
        memory: Optional[DetectionMemory] = None,
    ) -> str:
        """Build the analysis prompt for the LLM."""
        # Sample data for context
        sample_size = min(20, len(series))
        sample_indices = np.linspace(0, len(series) - 1, sample_size, dtype=int)
        sampled = [(str(series.index[i]), float(series.iloc[i])) for i in sample_indices]

        # Format potential anomalies
        anomaly_candidates = [
            (str(idx), float(val), float((val - series.mean()) / series.std()))
            for idx, val in potential_anomalies.items()
        ]

        prompt = f"""Analyze this time series for anomalies.

DATA STATISTICS:
- Total points: {len(series)}
- Mean: {series.mean():.2f}
- Std Dev: {series.std():.2f}
- Min: {series.min():.2f}
- Max: {series.max():.2f}
- Date range: {series.index[0]} to {series.index[-1]}

SAMPLED DATA (timestamp, value):
{sampled}

POTENTIAL ANOMALY CANDIDATES (timestamp, value, z_score):
{anomaly_candidates}

SENSITIVITY LEVEL: {sensitivity} (higher = fewer anomalies flagged)
- For sensitivity 2.0: flag clear outliers only
- For sensitivity 1.5: include moderate deviations
- For sensitivity 3.0: only extreme outliers"""

        # Inject memory context when available
        if memory is not None:
            if memory.baseline_stats:
                prompt += f"\n\nBASELINE STATISTICS (from a known-normal reference period):\n{memory.baseline_stats}"
                prompt += "\nUse these baseline stats to judge whether candidates truly deviate from normal behaviour."

            if memory.expected_range and len(memory.expected_range) == 2:
                prompt += (
                    f"\n\nEXPECTED VALUE RANGE: [{memory.expected_range[0]}, {memory.expected_range[1]}]"
                    "\nValues inside this range should generally NOT be flagged as anomalies."
                )

            if memory.context:
                prompt += f"\n\nDOMAIN CONTEXT (from the caller):\n{memory.context}"
                prompt += "\nIncorporate this domain knowledge into your anomaly assessment."

        prompt += """

Analyze the candidates above and return the genuine anomalies as JSON.
Consider the context and patterns, not just absolute values."""

        return prompt

    def _parse_response(self, response: str, sensitivity: float) -> Optional[dict]:
        """Parse LLM response to extract anomalies."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))

                anomalies = []
                if "anomalies" in parsed and isinstance(parsed["anomalies"], list):
                    for a in parsed["anomalies"]:
                        if not isinstance(a, dict):
                            continue

                        # Filter by confidence based on sensitivity
                        confidence = a.get("confidence", 0.5)
                        min_confidence = 0.3 + (sensitivity - 1.0) * 0.15

                        if confidence >= min_confidence:
                            anomaly_type_str = a.get("type", "spike").lower()
                            if anomaly_type_str == "drop":
                                anomaly_type = AnomalyType.DROP
                            elif anomaly_type_str == "shift":
                                anomaly_type = AnomalyType.SHIFT
                            else:
                                anomaly_type = AnomalyType.SPIKE

                            anomalies.append(Anomaly(
                                timestamp=str(a.get("timestamp", "")),
                                value=float(a.get("value", 0)),
                                score=float(confidence),
                                anomaly_type=anomaly_type
                            ))

                return {
                    "anomalies": anomalies,
                    "summary": parsed.get("summary", "")
                }

            return None

        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return None

    def _fallback_detection(
        self,
        series: pd.Series,
        sensitivity: float,
        mean_val: float,
        std_val: float
    ) -> DetectionResult:
        """Fallback to Z-score detection if LLM fails."""
        z_scores = (series - mean_val) / std_val
        anomaly_mask = np.abs(z_scores) > sensitivity

        anomalies = []
        for idx in series[anomaly_mask].index:
            val = float(series[idx])
            z = float(z_scores[idx])
            anomaly_type = AnomalyType.SPIKE if z > 0 else AnomalyType.DROP

            anomalies.append(Anomaly(
                timestamp=str(idx),
                value=val,
                score=abs(z),
                anomaly_type=anomaly_type
            ))

        return DetectionResult(
            success=True,
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            threshold=sensitivity,
            detector_name=f"{self.name}-fallback",
            metadata={"fallback_used": True, "mean": mean_val, "std": std_val}
        )
