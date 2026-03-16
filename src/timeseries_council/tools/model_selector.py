# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
LLM-driven model selection for anomaly detection and forecasting.

When an LLM provider is available, this module asks the LLM to choose
which detectors or forecasters to use based on the statistical profile
of the time series. Falls back to None (caller uses static priority)
if the LLM call fails or no provider is given.
"""

import json
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)


def compute_series_profile(series: pd.Series) -> dict:
    """Compute statistical profile of a time series for the LLM prompt.

    Returns dict with: length, mean, std, min, max, range, cv, acf1,
    skewness, kurtosis.
    """
    values = series.values.astype(float)
    std = float(np.std(values))
    mean = float(np.mean(values))

    # Autocorrelation lag-1
    if len(values) > 1:
        shifted = values[1:]
        original = values[:-1]
        if np.std(original) > 0 and np.std(shifted) > 0:
            acf1 = float(np.corrcoef(original, shifted)[0, 1])
        else:
            acf1 = 0.0
    else:
        acf1 = 0.0

    return {
        "length": len(values),
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "cv": std / abs(mean) if abs(mean) > 1e-10 else 0.0,
        "acf1": acf1,
        "skewness": float(pd.Series(values).skew()),
        "kurtosis": float(pd.Series(values).kurtosis()),
    }


_SELECTION_PROMPT = """\
You are a time series analysis expert. Given the characteristics of a
time series, select the 3-5 best {task_type} models from the available options.

Available models:
{model_list}

Series characteristics:
- Length: {length} points
- Mean: {mean:.4f}, Std: {std:.4f}
- Min: {min:.4f}, Max: {max:.4f}
- Range: {range:.4f}
- Coefficient of variation: {cv:.4f}
- Autocorrelation (lag-1): {acf1:.4f}
- Skewness: {skewness:.4f}
- Kurtosis: {kurtosis:.4f}

Return ONLY a JSON object with no other text: {{"models": ["name1", "name2", ...]}}
Select 3-5 models. Consider:
- Mix different method families for diversity
- Data length affects which methods work (short series → simpler methods)
- High autocorrelation suggests temporal structure
- High coefficient of variation suggests large spread → robust methods
- High kurtosis suggests heavy tails
"""


def llm_select_models(
    series: pd.Series,
    available: List[str],
    model_descriptions: Dict[str, str],
    provider,
    task_type: str = "anomaly detection",
    max_models: int = 5,
) -> Optional[List[str]]:
    """Use the LLM to select models based on series characteristics.

    Args:
        series: The time series data.
        available: List of available model names.
        model_descriptions: Dict mapping model name to one-line description.
        provider: LLM provider with .generate() method, or None.
        task_type: "anomaly detection" or "forecasting".
        max_models: Maximum number of models to select.

    Returns:
        List of selected model names, or None if LLM selection failed
        (caller should fall back to static priority).
    """
    if provider is None:
        return None

    # Build model list for prompt
    model_lines = []
    for name in available:
        desc = model_descriptions.get(name, name)
        model_lines.append(f"- {name}: {desc}")
    model_list = "\n".join(model_lines)

    # Compute series profile
    profile = compute_series_profile(series)

    prompt = _SELECTION_PROMPT.format(
        task_type=task_type,
        model_list=model_list,
        **profile,
    )

    try:
        response = provider.generate(prompt, temperature=0.1)
    except Exception as e:
        logger.warning(f"LLM model selection failed: {e}")
        return None

    # Parse response
    selected = _parse_model_response(response)
    if not selected:
        logger.warning("LLM returned unparseable model selection response")
        return None

    # Validate against available models
    valid = [m for m in selected if m in available]
    if len(valid) < 3:
        logger.warning(
            f"LLM selected too few valid models ({valid}), "
            f"requested: {selected}, available: {available}"
        )
        return None

    logger.info(f"LLM selected {task_type} models: {valid[:max_models]}")
    return valid[:max_models]


def _parse_model_response(response: str) -> List[str]:
    """Parse LLM response to extract model names."""
    # Try JSON in code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            names = data.get("models") or data.get("detectors") or data.get("forecasters")
            if isinstance(names, list):
                return [n.strip().lower() for n in names]
        except json.JSONDecodeError:
            pass

    # Try raw JSON with "models"/"detectors"/"forecasters" key
    for key in ("models", "detectors", "forecasters"):
        m = re.search(r'\{[^{}]*"' + key + r'"[^{}]*\}', response, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data.get(key), list):
                    return [n.strip().lower() for n in data[key]]
            except json.JSONDecodeError:
                pass

    # Try bare JSON array
    m = re.search(r'\[.*?\]', response, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(0))
            if isinstance(items, list) and all(isinstance(i, str) for i in items):
                return [n.strip().lower() for n in items]
        except json.JSONDecodeError:
            pass

    return []
