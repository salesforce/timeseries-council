# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Anthropic Claude LLM provider.
"""

from typing import Optional
from .base import BaseLLMProvider
from ..logging import get_logger
from ..exceptions import ProviderError

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-sonnet-4-20250514)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
        logger.info(f"Initialized Anthropic provider with model: {model}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """Generate text using Claude API."""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
            "temperature": temperature,
        }

        if system_instruction:
            kwargs["system"] = system_instruction

        logger.debug(f"Generating response with temperature={temperature}")

        try:
            response = self.client.messages.create(**kwargs)
            result = response.content[0].text
            logger.debug(f"Generated response: {len(result)} chars")
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "authentication" in error_msg:
                logger.error("Invalid Anthropic API key")
                raise ProviderError("Invalid Anthropic API key", provider="anthropic")
            elif "429" in error_msg or "rate" in error_msg:
                logger.error("Anthropic API rate limit exceeded")
                raise ProviderError("Anthropic API rate limit exceeded", provider="anthropic")
            logger.error(f"Anthropic API error: {e}")
            raise ProviderError(f"Anthropic API error: {e}", provider="anthropic")
