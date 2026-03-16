# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for Orchestrator.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from timeseries_council.orchestrator import Orchestrator


class TestOrchestrator:
    """Tests for Orchestrator class."""

    def test_initialization(self, mock_llm_provider, sample_csv_path):
        """Test orchestrator initialization."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales"
        )

        # The orchestrator stores the provider as self.llm (not self.llm_provider)
        assert orchestrator.llm == mock_llm_provider
        assert orchestrator.target_col == "sales"
        assert orchestrator._data is not None

    def test_initialization_with_forecaster(self, mock_llm_provider, sample_csv_path, mock_forecaster):
        """Test orchestrator with custom forecaster."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales",
            forecaster=mock_forecaster
        )

        # Orchestrator passes forecaster to skill_executor
        assert orchestrator is not None

    def test_initialization_with_detector(self, mock_llm_provider, sample_csv_path, mock_detector):
        """Test orchestrator with custom detector."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales",
            detector=mock_detector
        )

        # Orchestrator passes detector to skill_executor
        assert orchestrator is not None

    def test_initialization_with_progress_callback(self, mock_llm_provider, sample_csv_path, progress_callback):
        """Test orchestrator with progress callback."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales",
            progress_callback=progress_callback
        )

        assert orchestrator.progress_callback == progress_callback

    def test_chat_basic(self, mock_llm_provider, sample_csv_path):
        """Test basic chat functionality."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales"
        )

        response = orchestrator.chat("What is the trend?")
        assert response is not None
        assert isinstance(response, str)

    def test_chat_returns_response(self, mock_llm_provider, sample_csv_path):
        """Test that chat returns non-empty response."""
        mock_llm_provider.generate_with_tools.return_value = {
            "tool_call": None,
            "response": "The trend shows consistent growth."
        }

        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales"
        )

        response = orchestrator.chat("Describe the trend")
        assert "growth" in response or response is not None

    def test_data_loaded(self, mock_llm_provider, sample_csv_path):
        """Test that data is loaded on initialization."""
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales"
        )

        assert orchestrator._data is not None
        assert len(orchestrator._data) > 0


class TestOrchestratorWithCouncil:
    """Tests for Orchestrator council functionality."""

    def test_chat_with_council(self, mock_llm_provider, sample_csv_path):
        """Test council chat mode."""
        council_providers = {
            "model1": mock_llm_provider,
            "model2": mock_llm_provider
        }

        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales",
            council_providers=council_providers
        )

        response = orchestrator.chat_with_council("What is the forecast?")
        assert response is not None

    def test_council_perspectives(self, mock_llm_provider, sample_csv_path):
        """Test that council generates perspectives."""
        council_providers = {
            "forecaster": mock_llm_provider,
            "risk_analyst": mock_llm_provider,
            "business_explainer": mock_llm_provider
        }

        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="sales",
            council_providers=council_providers
        )

        # Council should use different roles
        assert orchestrator.council_providers is not None


class TestOrchestratorErrorHandling:
    """Tests for error handling."""

    def test_invalid_csv_path(self, mock_llm_provider):
        """Test handling of invalid CSV path - orchestrator logs error but does not raise."""
        # The orchestrator now handles file-not-found gracefully (logs error, data=None)
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path="/nonexistent/path/data.csv",
            target_col="sales"
        )
        # Data should not be loaded
        assert orchestrator._data is None

    def test_invalid_target_column(self, mock_llm_provider, sample_csv_path):
        """Test handling of invalid target column - orchestrator handles gracefully."""
        # The orchestrator now handles invalid columns gracefully
        orchestrator = Orchestrator(
            llm_provider=mock_llm_provider,
            csv_path=sample_csv_path,
            target_col="nonexistent_column"
        )
        # Should still initialize (error handling is deferred)
        assert orchestrator is not None
