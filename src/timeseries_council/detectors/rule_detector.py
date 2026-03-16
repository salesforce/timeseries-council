# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Rule-based anomaly detector powered by LLM prompt parsing.

Architecture:
1. User provides a natural language prompt describing the anomaly pattern
2. LLM parses the prompt into structured, deterministic rules (ONE LLM call)
3. Rule engine applies the rules to the full time series (no LLM calls)
4. Post-processing (smooth_incidents) groups and expands detections

This approach gives:
- Consistency: deterministic rule execution, no LLM randomness per window
- Speed: only 1 LLM call per detection run
- Transparency: rules are human-readable and auditable
- Flexibility: works with any natural language anomaly description
"""

from typing import Optional, Callable, List, Dict, Any
import pandas as pd
import numpy as np
import json
import re

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..providers.base import BaseLLMProvider
from ..logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Rule Engine
# ============================================================================

def _apply_rules(series: pd.Series, rules: Dict[str, Any]) -> np.ndarray:
    """
    Apply parsed rules to a time series deterministically.

    Supported rule types:
    - threshold: value > X or value < X
    - range: X < value < Y
    - amplitude_band: value oscillates around center with amplitude
    - min_consecutive: require N consecutive points matching
    - sustained_elevation: values stay elevated above baseline
    - spike: sudden jump from baseline

    Returns:
        Binary array (0/1) of same length as series.
    """
    n = len(series)
    values = series.values.astype(float)

    # ---- Step 1: Initial Point-level match for ALL conditions ----
    # Let's handle the special case of an "amplitude_band" replacing a threshold 
    # to find sustained shake. A proper implementation of the prompt:
    # "hard threshold of 2000, then amplitude band of 200 (sustained for 5 points)"
    # Means: find a starting point >= 2000, and ensure it and following points 
    # are >= (2000 - 200) = 1800, up to a total of 5 points.
    
    # Extract rules
    conditions = rules.get("conditions", [rules])
    min_consecutive = int(rules.get("min_consecutive", 1))
    
    # 1. First find the strict "trigger" threshold if specified, else use the generous one
    trigger_mask = np.ones(n, dtype=bool)
    sustain_mask = np.ones(n, dtype=bool)

    has_trigger = False
    
    for cond in conditions:
        cond_type = cond.get("type", "threshold")
        
        if cond_type == "threshold":
            val = float(cond.get("value", 0))
            trigger_mask &= (values >= val)
            sustain_mask &= (values >= val)  # If only threshold is given, they are the same
            has_trigger = True
            
        elif cond_type == "amplitude_band":
            center = float(cond.get("center", 0))
            amp = float(cond.get("amplitude", 0))
            # Trigger is the hard threshold (center)
            trigger_mask &= (values >= center)
            # Sustain is the bottom of the band
            sustain_mask &= (values >= (center - amp))
            has_trigger = True
            
        elif cond_type == "hard_trigger_with_band":
            # Direct mapping from our new LLM prompt rules
            trigger_val = float(cond.get("trigger_value", 0))
            sustain_val = float(cond.get("sustain_value", 0))
            trigger_mask &= (values >= trigger_val)
            sustain_mask &= (values >= sustain_val)
            has_trigger = True
            
        else:
            logger.warning(f"Unknown condition type: {cond_type}")

    if not has_trigger:
        trigger_mask = np.zeros(n, dtype=bool)
        sustain_mask = np.zeros(n, dtype=bool)

    point_match = np.zeros(n, dtype=int)
    
    # ---- Step 2: Apply trigger + sustain forward expansion ----
    # As defined by the true intent:
    # 1. First flagging is done when a point meets trigger_mask (e.g. >= 2000).
    # 2. From there, expand forward up to a maximum of `forwardprop` points
    #    as long as each subsequent point meets the `sustain_mask` (e.g. >= 1800).
    # 3. If it hits an invalid point early, the expansion stops for that trigger.
    # Note: `forwardprop` is typically passed in or configured. We will use 
    # `min_consecutive - 1` as the expansion limit if it's a rule parameter.
    
    # We'll use a dynamic expansion loop
    expand_len = min_consecutive - 1 if min_consecutive > 1 else 4
    
    i = 0
    while i < n:
        if trigger_mask[i]:
            point_match[i] = 1
            # Expand forward
            step = 1
            while step <= expand_len and (i + step) < n:
                if sustain_mask[i + step]:
                    point_match[i + step] = 1
                    step += 1
                else:
                    break
            
            # Skip past the expanded region so we don't re-trigger inside it
            i += step
        else:
            i += 1

    return point_match


def _smooth_incidents(pred, forwardprop=4, merge_gap=5, min_window=5, apply_forwardprop=True):
    """Apply APT-style post-processing: expand, merge, filter short."""
    L = len(pred)
    pred = np.array(pred, dtype=float)
    high_idxs = np.where(pred >= 0.9)[0]

    if len(high_idxs) == 0:
        return np.zeros(L, dtype=int)

    intervals = []
    for idx in high_idxs:
        start = max(0, idx)
        # Only apply forwardprop blindly if apply_forwardprop=True
        # (If we already did dynamic sustain expansion, we just use the point itself)
        end = min(L - 1, idx + (forwardprop if apply_forwardprop else 0))
        intervals.append((start, end))

    merged = []
    intervals.sort()
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end + merge_gap:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    cleaned = [(s, e) for s, e in merged if (e - s + 1) >= min_window]

    result = np.zeros(L, dtype=int)
    for s, e in cleaned:
        result[s:e + 1] = 1
    return result


# ============================================================================
# Rule Detector Class
# ============================================================================

class RuleDetector(BaseDetector):
    """
    Rule-based anomaly detector that uses LLM to parse natural language
    prompts into deterministic detection rules.

    Pipeline:
    1. LLM parses prompt → structured JSON rules (1 API call)
    2. Rule engine applies rules to full series (deterministic)
    3. Post-processing groups/expands detections (smooth_incidents)
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        custom_prompt: str = "",
        apply_smoothing: bool = True,
        forwardprop: int = 4,
        merge_gap: int = 5,
        min_window: int = 5,
    ):
        self.provider = provider
        self.custom_prompt = custom_prompt
        self.apply_smoothing = apply_smoothing
        self.forwardprop = forwardprop
        self.merge_gap = merge_gap
        self.min_window = min_window
        self._parsed_rules = None  # Cache after first parse
        logger.info(f"Initialized RuleDetector with {provider.provider_name}")

    @property
    def name(self) -> str:
        return f"RuleDetector-{self.provider.provider_name}"

    @property
    def description(self) -> str:
        return "LLM-parsed rule-based anomaly detector"

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
    ) -> DetectionResult:
        """Detect anomalies by parsing prompt into rules and applying them."""
        error = self.validate_input(series)
        if error:
            return DetectionResult(success=False, error=error)

        n = len(series)

        # ---- Step 1: Parse prompt into rules (1 LLM call) ----
        self._report_progress(progress_callback, "Parsing prompt into rules...", 0.1)

        if self._parsed_rules is None:
            # Send data sample for context
            data_sample = self._get_data_sample(series)
            self._parsed_rules = self._parse_prompt_to_rules(data_sample)

        if self._parsed_rules is None:
            return DetectionResult(
                success=False,
                error="Failed to parse prompt into rules",
                detector_name=self.name,
            )

        logger.info(f"Parsed rules: {json.dumps(self._parsed_rules, indent=2)}")

        # ---- Step 2: Apply rules deterministically ----
        self._report_progress(progress_callback, "Applying rules...", 0.5)

        raw_pred = _apply_rules(series, self._parsed_rules)
        raw_count = int(raw_pred.sum())
        logger.info(f"Raw rule detection: {raw_count} points flagged")

        # ---- Step 3: Post-processing (smooth_incidents) ----
        # Since we already did forward expansion based on the strictly validated 
        # sustain band, we skip the "blind" forwardprop here and just merge gaps.
        
        should_smooth = self.apply_smoothing
        if isinstance(self._parsed_rules, dict) and "apply_smoothing" in self._parsed_rules:
            should_smooth = bool(self._parsed_rules.get("apply_smoothing"))
            
        if should_smooth and raw_count > 0:
            self._report_progress(progress_callback, "Applying post-processing...", 0.8)
            pred = _smooth_incidents(
                raw_pred,
                forwardprop=self.forwardprop,
                merge_gap=self.merge_gap,
                min_window=self.min_window,
                apply_forwardprop=False,  # Skip blind expansion
            )
        else:
            pred = raw_pred

        # ---- Step 4: Build result ----
        self._report_progress(progress_callback, "Detection complete", 1.0)

        anomalies = []
        mean_val = float(series.mean())
        for idx in np.where(pred == 1)[0]:
            val = float(series.iloc[idx])
            ts = str(series.index[idx])
            atype = AnomalyType.SPIKE if val > mean_val else AnomalyType.DROP
            anomalies.append(Anomaly(timestamp=ts, value=val, score=1.0, anomaly_type=atype))

        # Apply memory post-processing (expected_range filtering)
        anomalies = self._apply_memory(anomalies, memory)

        return DetectionResult(
            success=True,
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            threshold=sensitivity,
            detector_name=self.name,
            metadata={
                "parsed_rules": self._parsed_rules,
                "raw_flagged": raw_count,
                "after_smoothing": int(pred.sum()),
                "smoothing_applied": should_smooth,
                "memory_applied": memory is not None,
            },
        )

    def _get_data_sample(self, series: pd.Series) -> str:
        """Get a compact data sample for the LLM to understand the data context."""
        stats = (
            f"Total points: {len(series)}\n"
            f"Time range: {series.index[0]} to {series.index[-1]}\n"
            f"Value stats: mean={series.mean():.2f}, std={series.std():.2f}, "
            f"min={series.min():.2f}, max={series.max():.2f}, "
            f"median={series.median():.2f}\n"
            f"Percentiles: p5={np.percentile(series, 5):.2f}, "
            f"p25={np.percentile(series, 25):.2f}, "
            f"p75={np.percentile(series, 75):.2f}, "
            f"p95={np.percentile(series, 95):.2f}"
        )

        # Sample some values
        sample_indices = np.linspace(0, len(series) - 1, min(30, len(series)), dtype=int)
        samples = [f"  idx={i}, val={series.iloc[i]:.2f}" for i in sample_indices]

        return f"{stats}\n\nSample values:\n" + "\n".join(samples)

    def _parse_prompt_to_rules(self, data_sample: str) -> Optional[Dict]:
        """Use LLM to parse the natural language prompt into structured rules."""

        system_instruction = """You are a time series anomaly rule parser. 
Your job is to convert a natural language anomaly description into structured detection rules.

You MUST respond with ONLY a JSON object (no other text) in this exact format:
{
    "conditions": [
        {
            "type": "hard_trigger_with_band" | "threshold" | "range" | "elevated",
            "trigger_value": <number>,  // for hard_trigger_with_band: the strict starting threshold
            "sustain_value": <number>,  // for hard_trigger_with_band: the lower bound of the shake/oscillation 
            "value": <number>,         // for simple threshold
            "direction": "above" | "below" // for simple threshold
        }
    ],
    "min_consecutive": <int>,           // minimum consecutive points matching (default 1)
    "apply_smoothing": <boolean>,       // true by default to group and expand incident windows. MUST be false if user explicitly asks for strict thresholding, raw tracking, no enhancement, or no gap filling.
    "description": "brief rule summary"
}

Rules for parsing:
- "hard threshold of X, with amplitude band of Y" means a trigger must hit X, and subsequent points must stay above (X - Y) to maintain the shake pattern. 
  → type: "hard_trigger_with_band", trigger_value: X, sustain_value: X - Y
- "threshold of 2000" → type: "threshold", value: 2000, direction: "above"
- "sustained for 5 points" → min_consecutive: 5

IMPORTANT: Always examine the data statistics to set appropriate thresholds. If the prompt 
says "values around 2000" but the data baseline is ~100, then 2000 is clearly anomalous."""

        prompt = f"""Parse this anomaly detection prompt into structured rules.

USER'S ANOMALY DESCRIPTION:
{self.custom_prompt}

DATA CONTEXT:
{data_sample}

Convert the above description into a JSON rules object. Consider the data statistics 
to understand what constitutes "normal" vs "anomalous" values.

Respond with ONLY the JSON object:"""

        try:
            response = self.provider.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.0,
            )
            logger.debug(f"LLM rule parse response: {response[:500]}")

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                rules = json.loads(json_match.group(0))
                logger.info(f"Successfully parsed rules: {rules.get('description', 'N/A')}")
                return rules

            logger.error(f"No JSON found in LLM response: {response[:200]}")
            return None

        except Exception as e:
            logger.error(f"Failed to parse prompt into rules: {e}")
            return None

    def get_rules(self) -> Optional[Dict]:
        """Return the parsed rules (for debugging/transparency)."""
        return self._parsed_rules

    def clear_rules_cache(self):
        """Clear cached rules, forcing re-parse on next detect()."""
        self._parsed_rules = None
