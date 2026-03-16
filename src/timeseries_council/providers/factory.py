# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Provider factory for creating LLM provider instances.
"""

from typing import Dict, Type, Optional, List
from .base import BaseLLMProvider
from ..logging import get_logger
from ..exceptions import ProviderError

logger = get_logger(__name__)


# Registry of available providers (populated lazily)
_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {}


def _get_providers() -> Dict[str, Type[BaseLLMProvider]]:
    """Lazily load provider classes to avoid import errors."""
    global _PROVIDERS
    if not _PROVIDERS:
        from .gemini import GeminiProvider
        from .anthropic import AnthropicProvider
        from .openai_provider import OpenAIProvider
        from .deepseek import DeepSeekProvider
        from .qwen import QwenProvider

        _PROVIDERS = {
            "gemini": GeminiProvider,
            "anthropic": AnthropicProvider,
            "claude": AnthropicProvider,  # alias
            "openai": OpenAIProvider,
            "gpt": OpenAIProvider,  # alias
            "deepseek": DeepSeekProvider,
            "qwen": QwenProvider,
        }
    return _PROVIDERS


def create_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Factory function to create LLM provider instances.

    Args:
        provider_name: Name of the provider (gemini, anthropic, openai, deepseek, qwen)
        api_key: API key for the provider
        model: Optional model name override
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ProviderError: If provider_name is not recognized
    """
    providers = _get_providers()
    provider_name = provider_name.lower().strip()

    if provider_name not in providers:
        available = get_available_providers()
        logger.warning(f"Unknown provider: {provider_name}")
        return None

    provider_class = providers[provider_name]
    logger.info(f"Creating provider: {provider_name}")

    if model:
        return provider_class(api_key=api_key, model=model, **kwargs)

    return provider_class(api_key=api_key, **kwargs)


def get_available_providers() -> List[str]:
    """Return list of available provider names (excluding aliases)."""
    providers = _get_providers()
    return [k for k in providers.keys() if k not in ("claude", "gpt")]


# Alias for backwards compatibility
list_providers = get_available_providers
PROVIDERS = property(lambda self: _get_providers())
