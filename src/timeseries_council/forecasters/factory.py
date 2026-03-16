# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Factory for creating forecaster instances.
"""

from typing import Dict, Type, Optional, List, Any
from .base import BaseForecaster
from ..logging import get_logger
from ..exceptions import ForecasterError

logger = get_logger(__name__)


# Registry of available forecasters
_FORECASTERS: Dict[str, Type[BaseForecaster]] = {}


def _get_forecasters() -> Dict[str, Type[BaseForecaster]]:
    """Lazily load forecaster classes with graceful fallback."""
    global _FORECASTERS
    if not _FORECASTERS:
        from .zscore_baseline import ZScoreBaselineForecaster
        from .llm_forecaster import LLMForecaster

        # Start with always-available forecasters
        _FORECASTERS = {
            "zscore_baseline": ZScoreBaselineForecaster,
            "baseline": ZScoreBaselineForecaster,
            "llm": LLMForecaster,
        }

        # Try loading optional forecasters
        try:
            from .moirai import MoiraiForecaster
            _FORECASTERS["moirai"] = MoiraiForecaster
            logger.info("Moirai2 forecaster available")
        except ImportError as e:
            logger.warning(f"Moirai2 forecaster not available: {e}")

        try:
            from .chronos import ChronosForecaster
            _FORECASTERS["chronos"] = ChronosForecaster
            _FORECASTERS["chronos2"] = ChronosForecaster  # Alias for backwards compatibility
            logger.info("Chronos forecaster available")
        except ImportError as e:
            logger.warning(f"Chronos forecaster not available: {e}")

        try:
            from .timesfm import TimesFMForecaster
            _FORECASTERS["timesfm"] = TimesFMForecaster
            logger.info("TimesFM forecaster available")
        except ImportError as e:
            logger.warning(f"TimesFM forecaster not available: {e}")

        try:
            from .lag_llama import LagLlamaForecaster
            _FORECASTERS["lag-llama"] = LagLlamaForecaster
            _FORECASTERS["lagllama"] = LagLlamaForecaster
            logger.info("Lag-Llama forecaster available")
        except ImportError as e:
            logger.warning(f"Lag-Llama forecaster not available: {e}")

        try:
            from .tirex import TiRexForecaster
            _FORECASTERS["tirex"] = TiRexForecaster
            logger.info("TiRex forecaster available")
        except ImportError as e:
            logger.warning(f"TiRex forecaster not available: {e}")

    return _FORECASTERS


def create_forecaster(
    forecaster_name: str,
    auto_setup: bool = True,
    **kwargs: Any
) -> BaseForecaster:
    """
    Factory function to create forecaster instances.

    Args:
        forecaster_name: Name of the forecaster
            Options: 'moirai', 'chronos', 'timesfm', 'lag-llama', 'tirex', 'llm', 'baseline'
        auto_setup: Auto-install missing packages and download models (default True)
        **kwargs: Forecaster-specific arguments
            - moirai: model_size, model_variant, device
            - chronos: model_size, device
            - timesfm: model_size, device
            - lag-llama: device, context_length, use_rope_scaling
            - llm: provider (required BaseLLMProvider instance)

    Returns:
        Configured forecaster instance (falls back to baseline if unavailable)
    """
    forecasters = _get_forecasters()
    name = forecaster_name.lower().strip()

    # Try original name first, then hyphenated version
    if name not in forecasters:
        name = name.replace("_", "-")

    # Auto-setup if enabled and forecaster not available
    if name not in forecasters and auto_setup:
        try:
            from ..setup_models import setup_model
            logger.info(f"Attempting auto-setup for {name}")
            setup_result = setup_model(name, auto_install=True)
            if setup_result["success"]:
                logger.info(f"Auto-setup succeeded for {name}")
                # Clear the forecasters cache to reload
                global _FORECASTERS
                _FORECASTERS = {}
                forecasters = _get_forecasters()
            else:
                logger.warning(f"Auto-setup for {name} failed: {setup_result['message']}")
        except Exception as e:
            logger.warning(f"Auto-setup failed: {e}")

    if name not in forecasters:
        available = get_available_forecasters()
        logger.warning(f"Forecaster '{forecaster_name}' not available. Using baseline. Available: {available}")
        name = "baseline"

    forecaster_class = forecasters[name]
    logger.info(f"Creating forecaster: {name}")

    try:
        return forecaster_class(**kwargs)
    except Exception as e:
        logger.error(f"Failed to create {name} forecaster: {e}. Falling back to baseline.")
        return forecasters["baseline"]()


def get_available_forecasters() -> List[str]:
    """Return list of available forecaster names (excluding aliases)."""
    forecasters = _get_forecasters()
    return [k for k in forecasters.keys() if k not in ("lagllama", "baseline", "chronos2", "zscore_baseline")]


# Alias for backwards compatibility
list_forecasters = get_available_forecasters


def get_forecaster_info() -> Dict[str, Dict[str, Any]]:
    """Return information about all available forecasters."""
    return {
        "moirai": {
            "name": "Moirai",
            "description": "Salesforce Moirai foundation model via uni2ts",
            "sizes": ["small", "base", "large"],
            "requires": ["uni2ts", "gluonts", "torch"],
        },
        "chronos": {
            "name": "Chronos",
            "description": "Amazon Chronos foundation model",
            "sizes": ["base", "synth", "small"],
            "requires": ["chronos-forecasting"],
        },
        "timesfm": {
            "name": "TimesFM",
            "description": "Google TimesFM foundation model",
            "sizes": ["200m"],
            "requires": ["timesfm"],
        },
        "lag-llama": {
            "name": "Lag-Llama",
            "description": "Probabilistic time series foundation model",
            "sizes": ["default"],
            "requires": ["lag-llama", "gluonts"],
        },
        "tirex": {
            "name": "TiRex",
            "description": "NX-AI TiRex foundation model for quantile time series forecasting",
            "sizes": ["small", "base", "large"],
            "requires": ["tirex"],
        },
        "llm": {
            "name": "LLM Forecaster",
            "description": "Zero-shot forecasting using LLMs (Claude, GPT, Gemini)",
            "sizes": ["n/a"],
            "requires": ["provider instance"],
        },
    }
