# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tests for tools implementations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from timeseries_council.tools import TOOLS
from timeseries_council.tools.registry import ToolRegistry


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_registry_has_tools(self):
        """Test that registry contains expected tools."""
        expected_tools = [
            "run_forecast",
            "describe_series",
            "detect_anomalies",
            "what_if_simulation",
            "decompose_series",
            "compare_series"
        ]

        for tool_name in expected_tools:
            assert tool_name in TOOLS, f"Tool {tool_name} should be in registry"

    def test_tool_has_required_fields(self):
        """Test that each tool has required fields."""
        required_fields = ["description", "parameters", "function"]

        for tool_name, tool_def in TOOLS.items():
            for field in required_fields:
                assert field in tool_def, f"Tool {tool_name} missing {field}"

    def test_tool_parameters_are_dict(self):
        """Test that tool parameters are properly formatted as dicts."""
        for tool_name, tool_def in TOOLS.items():
            params = tool_def["parameters"]
            assert isinstance(params, dict), f"Tool {tool_name} parameters should be a dict"

    def test_all_expected_tools_present(self):
        """Test that all expected tools are in the registry."""
        expected = [
            "run_forecast", "describe_series", "detect_anomalies",
            "what_if_simulation", "decompose_series", "compare_series",
            "sensitivity_analysis", "backtest_forecast", "compare_periods",
        ]
        for tool in expected:
            assert tool in TOOLS, f"Tool {tool} should be in TOOLS registry"


class TestForecastTool:
    """Tests for forecast tool."""

    def test_forecast_tool_exists(self):
        """Test that run_forecast tool exists."""
        assert "run_forecast" in TOOLS

    def test_forecast_tool_definition(self):
        """Test forecast tool definition."""
        tool = TOOLS["run_forecast"]
        assert "horizon" in tool["parameters"] or "csv_path" in tool["parameters"]
        assert callable(tool["function"])


class TestDescribeTool:
    """Tests for describe_series tool."""

    def test_describe_tool_exists(self):
        """Test that describe tool exists."""
        assert "describe_series" in TOOLS

    def test_describe_tool_callable(self):
        """Test describe tool is callable."""
        tool = TOOLS["describe_series"]
        assert callable(tool["function"])


class TestAnomalyTool:
    """Tests for detect_anomalies tool."""

    def test_anomaly_tool_exists(self):
        """Test that anomaly tool exists."""
        assert "detect_anomalies" in TOOLS

    def test_anomaly_tool_definition(self):
        """Test anomaly tool definition."""
        tool = TOOLS["detect_anomalies"]
        params = tool["parameters"]
        assert "sensitivity" in params or "csv_path" in params


class TestWhatIfTool:
    """Tests for what_if_simulation tool."""

    def test_whatif_tool_exists(self):
        """Test that what_if tool exists."""
        assert "what_if_simulation" in TOOLS

    def test_whatif_tool_definition(self):
        """Test what_if tool definition."""
        tool = TOOLS["what_if_simulation"]
        params = tool["parameters"]
        assert "change_percent" in params or "scenario" in params or "csv_path" in params


class TestDecomposeTool:
    """Tests for decompose_series tool."""

    def test_decompose_tool_exists(self):
        """Test that decompose tool exists."""
        assert "decompose_series" in TOOLS


class TestCompareTool:
    """Tests for compare_series tool."""

    def test_compare_tool_exists(self):
        """Test that compare tool exists."""
        assert "compare_series" in TOOLS
