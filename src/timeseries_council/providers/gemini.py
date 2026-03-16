# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Google Gemini LLM provider.
"""

from typing import Optional
from .base import BaseLLMProvider
from ..logging import get_logger
from ..exceptions import ProviderError

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", **kwargs):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key
            model: Model name (default: gemini-2.5-flash)
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        logger.info(f"Initialized Gemini provider with model: {model}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """Generate text using Gemini API."""
        from google.genai import types

        config = types.GenerateContentConfig(temperature=temperature)
        if system_instruction:
            config.system_instruction = system_instruction

        logger.debug(f"Generating response with temperature={temperature}")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            logger.debug(f"Generated response: {len(response.text)} chars")
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in error_msg or "api key" in error_msg:
                logger.error("Invalid Gemini API key")
                raise ProviderError("Invalid Gemini API key", provider="gemini")
            elif "429" in error_msg or "quota" in error_msg:
                logger.error("Gemini API rate limit exceeded")
                raise ProviderError("Gemini API rate limit exceeded", provider="gemini")
            logger.error(f"Gemini API error: {e}")
            raise ProviderError(f"Gemini API error: {e}", provider="gemini")
