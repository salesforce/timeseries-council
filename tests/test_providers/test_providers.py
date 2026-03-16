# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for LLM provider implementations.
"""

import pytest
from unittest.mock import MagicMock, patch

from timeseries_council.providers.base import BaseLLMProvider
from timeseries_council.providers.factory import create_provider, list_providers


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_cannot_instantiate_abstract(self):
        """BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_subclass_must_implement_methods(self):
        """Subclass must implement abstract methods."""
        class IncompleteProvider(BaseLLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


class TestProviderFactory:
    """Tests for provider factory."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        assert isinstance(providers, list)
        # These are the expected providers
        expected = ["gemini", "anthropic", "openai", "deepseek", "qwen"]
        for p in expected:
            assert p in providers

    def test_create_provider_without_api_key(self):
        """Test that creating provider without API key handles gracefully."""
        # This should either return None or raise an error
        # depending on the implementation
        with patch.dict('os.environ', {}, clear=True):
            provider = create_provider("gemini")
            # Provider creation may fail without API key
            # Just verify it doesn't crash unexpectedly

    def test_create_unknown_provider(self):
        """Test creating unknown provider returns None."""
        provider = create_provider("nonexistent_provider")
        assert provider is None


class TestMockProvider:
    """Tests using mock provider."""

    def test_generate(self, mock_llm_provider):
        """Test generate method."""
        response = mock_llm_provider.generate("Test prompt")
        assert response == "This is a mock response."

    def test_generate_with_tools(self, mock_llm_provider):
        """Test generate_with_tools method."""
        result = mock_llm_provider.generate_with_tools("Test prompt", [])
        assert result["response"] == "Mock analysis complete."
        assert result["tool_call"] is None

    def test_provider_properties(self, mock_llm_provider):
        """Test provider properties."""
        assert mock_llm_provider.provider_name == "mock"
        assert mock_llm_provider.model_name == "mock-model"
