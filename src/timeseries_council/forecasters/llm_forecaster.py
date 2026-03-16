# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
LLM-based forecaster using language models.
"""

from typing import Optional, Callable
import pandas as pd
import json
import re

from .base import BaseForecaster
from ..types import ForecastResult
from ..providers.base import BaseLLMProvider
from ..logging import get_logger

logger = get_logger(__name__)


class LLMForecaster(BaseForecaster):
    """Forecaster using Large Language Models for zero-shot forecasting."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        include_context: bool = True
    ):
        """
        Initialize LLM forecaster.

        Args:
            provider: LLM provider to use (Claude, GPT, Gemini, etc.)
            include_context: Whether to include data statistics in prompt
        """
        self.provider = provider
        self.include_context = include_context
        logger.info(f"Initialized LLMForecaster with {provider.provider_name}")

    @property
    def name(self) -> str:
        return f"LLM-{self.provider.provider_name}"

    @property
    def description(self) -> str:
        return f"LLM-based forecaster using {self.provider.provider_name}"

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        context_length: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ForecastResult:
        """Generate forecast using LLM analysis."""
        import numpy as np

        error = self.validate_input(series, horizon)
        if error:
            logger.error(f"Validation failed: {error}")
            return ForecastResult(success=False, error=error)

        self._report_progress(progress_callback, "Preparing prompt...", 0.1)

        try:
            # Prepare data summary
            ctx_len = context_length or min(50, len(series))
            recent_data = series.tail(ctx_len)

            # Calculate statistics
            stats = {
                "mean": float(recent_data.mean()),
                "std": float(recent_data.std()),
                "min": float(recent_data.min()),
                "max": float(recent_data.max()),
                "last_value": float(series.iloc[-1]),
                "trend": self._calculate_trend(recent_data),
            }

            # Sample recent values for context
            sample_size = min(10, len(recent_data))
            sample_indices = np.linspace(0, len(recent_data) - 1, sample_size, dtype=int)
            sample_values = [round(float(recent_data.iloc[i]), 2) for i in sample_indices]

            self._report_progress(progress_callback, "Generating forecast with LLM...", 0.3)

            # Build prompt
            prompt = self._build_prompt(series, horizon, stats, sample_values)

            system_instruction = """You are an expert time series forecaster.
Analyze the provided data and generate accurate point forecasts.
Return your response as a valid JSON object with the following structure:
{
    "forecast": [list of numeric forecast values],
    "reasoning": "brief explanation of your forecast logic",
    "confidence": "high/medium/low"
}
IMPORTANT: Return ONLY the JSON object, no additional text."""

            # Call LLM
            response = self.provider.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.3
            )

            self._report_progress(progress_callback, "Parsing response...", 0.7)

            # Parse response
            result = self._parse_response(response, horizon)

            if result is None:
                logger.warning("Failed to parse LLM response, using fallback")
                # Fallback: simple trend extrapolation
                result = self._fallback_forecast(series, horizon, stats)

            self._report_progress(progress_callback, "Generating timestamps...", 0.9)

            # Generate timestamps - use robust frequency inference
            from ..utils import infer_frequency
            last_timestamp = series.index[-1]
            freq = infer_frequency(series)
            # Generate future dates starting after last timestamp using periods + 1 and skip first
            future_index = pd.date_range(
                start=last_timestamp,
                periods=horizon + 1,
                freq=freq
            )[1:]  # Skip the first one which is last_timestamp
            timestamps = [str(t) for t in future_index]

            self._report_progress(progress_callback, "Forecast complete", 1.0)

            logger.info(f"LLM forecast generated: {horizon} steps")

            return ForecastResult(
                success=True,
                forecast=result["forecast"],
                timestamps=timestamps,
                uncertainty=result.get("uncertainty"),
                horizon=horizon,
                model_name=f"llm-{self.provider.provider_name}",
                metadata={
                    "reasoning": result.get("reasoning", ""),
                    "confidence": result.get("confidence", "unknown"),
                    "context_length": ctx_len
                }
            )

        except Exception as e:
            logger.error(f"LLM forecast failed: {e}")
            return ForecastResult(success=False, error=str(e))

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate simple trend direction."""
        import numpy as np

        if len(series) < 3:
            return "unknown"

        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        std = series.std()

        if slope > 0.01 * std:
            return "increasing"
        elif slope < -0.01 * std:
            return "decreasing"
        else:
            return "stable"

    def _build_prompt(
        self,
        series: pd.Series,
        horizon: int,
        stats: dict,
        sample_values: list
    ) -> str:
        """Build the prompt for the LLM."""
        prompt = f"""Analyze this time series and forecast the next {horizon} values.

DATA STATISTICS:
- Data points: {len(series)}
- Mean: {stats['mean']:.2f}
- Std Dev: {stats['std']:.2f}
- Min: {stats['min']:.2f}
- Max: {stats['max']:.2f}
- Last value: {stats['last_value']:.2f}
- Trend: {stats['trend']}

RECENT VALUES (sampled):
{sample_values}

DATE RANGE:
- Start: {series.index[0]}
- End: {series.index[-1]}

Generate exactly {horizon} forecast values as a JSON object."""

        return prompt

    def _parse_response(self, response: str, horizon: int) -> Optional[dict]:
        """Parse LLM response to extract forecast."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))

                if "forecast" in parsed and isinstance(parsed["forecast"], list):
                    forecast = [float(v) for v in parsed["forecast"][:horizon]]

                    # Pad if needed
                    while len(forecast) < horizon:
                        forecast.append(forecast[-1] if forecast else 0)

                    return {
                        "forecast": [round(v, 2) for v in forecast],
                        "reasoning": parsed.get("reasoning", ""),
                        "confidence": parsed.get("confidence", "unknown")
                    }

            logger.warning(f"Could not parse forecast from response: {response[:200]}")
            return None

        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return None

    def _fallback_forecast(
        self,
        series: pd.Series,
        horizon: int,
        stats: dict
    ) -> dict:
        """Simple fallback forecast using trend extrapolation."""
        import numpy as np

        recent = series.tail(7)
        if len(recent) >= 2:
            daily_change = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        else:
            daily_change = 0

        last_value = stats["last_value"]
        forecast = [
            round(last_value + daily_change * (i + 1), 2)
            for i in range(horizon)
        ]

        # Simple uncertainty based on std
        uncertainty = [round(stats["std"], 2)] * horizon

        return {
            "forecast": forecast,
            "uncertainty": uncertainty,
            "reasoning": "Fallback: linear trend extrapolation",
            "confidence": "low"
        }
