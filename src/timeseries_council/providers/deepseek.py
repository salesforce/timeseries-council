# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
DeepSeek LLM provider (OpenAI-compatible API).
"""

from typing import Optional
from .base import BaseLLMProvider
from ..logging import get_logger
from ..exceptions import ProviderError

logger = get_logger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider (uses OpenAI-compatible endpoint)."""

    provider_name = "deepseek"

    def __init__(self, api_key: str, model: str = "deepseek-chat", **kwargs):
        """
        Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Model name (default: deepseek-chat)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        base_url = kwargs.get("base_url", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model
        logger.info(f"Initialized DeepSeek provider with model: {model}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """Generate text using DeepSeek API."""
        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating response with temperature={temperature}")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            result = response.choices[0].message.content
            logger.debug(f"Generated response: {len(result)} chars")
            return result
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "authentication" in error_msg:
                logger.error("Invalid DeepSeek API key")
                raise ProviderError("Invalid DeepSeek API key", provider="deepseek")
            elif "429" in error_msg or "rate" in error_msg:
                logger.error("DeepSeek API rate limit exceeded")
                raise ProviderError("DeepSeek API rate limit exceeded", provider="deepseek")
            logger.error(f"DeepSeek API error: {e}")
            raise ProviderError(f"DeepSeek API error: {e}", provider="deepseek")
