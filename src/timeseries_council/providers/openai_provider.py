# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
OpenAI LLM provider.
"""

from typing import Optional
from .base import BaseLLMProvider
from ..logging import get_logger
from ..exceptions import ProviderError

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model
        logger.info(f"Initialized OpenAI provider with model: {model}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """Generate text using OpenAI API."""
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
                logger.error("Invalid OpenAI API key")
                raise ProviderError("Invalid OpenAI API key", provider="openai")
            elif "429" in error_msg or "rate" in error_msg:
                logger.error("OpenAI API rate limit exceeded")
                raise ProviderError("OpenAI API rate limit exceeded", provider="openai")
            logger.error(f"OpenAI API error: {e}")
            raise ProviderError(f"OpenAI API error: {e}", provider="openai")
