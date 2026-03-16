# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Abstract base class for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import re
import json

from ..logging import get_logger

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for display."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Generate text response from the LLM.

        Args:
            prompt: The user prompt/message
            system_instruction: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text response
        """
        pass

    def parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON tool call from LLM response.
        Shared implementation across all providers.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed tool call dict or None if not found
        """
        # Try to find JSON in code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                logger.debug(f"Parsed tool call from code block: {result.get('tool')}")
                return result
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object with "tool" key
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                logger.debug(f"Parsed tool call from raw JSON: {result.get('tool')}")
                return result
            except json.JSONDecodeError:
                pass

        # Try to find any JSON-like structure
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "tool" in parsed:
                    logger.debug(f"Parsed tool call from JSON structure: {parsed.get('tool')}")
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.debug("No tool call found in response")
        return None
