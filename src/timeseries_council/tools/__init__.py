# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tools package for the orchestrator.

Provides tools that the LLM can call:
- run_forecast: Generate forecasts using various models
- describe_series: Get statistical summary
- decompose_series: Decompose into trend/seasonal/residual
- compare_series: Compare multiple columns
- detect_anomalies: Find anomalies using various detectors
- what_if_simulation: Scenario analysis
- sensitivity_analysis: Parameter sensitivity testing
- backtest_forecast: Test forecasts with custom windows and compare with actuals
"""

from .registry import (
    ToolRegistry,
    get_registry,
    register_tool,
    get_tools
)

from .forecasting import run_forecast, TOOL_INFO as FORECAST_TOOL
from .analysis import (
    describe_series,
    decompose_series,
    compare_series,
    compare_periods,
    DESCRIBE_TOOL,
    DECOMPOSE_TOOL,
    COMPARE_TOOL,
    COMPARE_PERIODS_TOOL
)
from .anomaly import detect_anomalies, TOOL_INFO as ANOMALY_TOOL
from .simulation import (
    what_if_simulation,
    sensitivity_analysis,
    WHAT_IF_TOOL,
    SENSITIVITY_TOOL
)
from .backtesting import backtest_forecast, TOOL_INFO as BACKTEST_TOOL


def register_all_tools() -> ToolRegistry:
    """Register all standard tools and return the registry."""
    registry = get_registry()

    # Forecasting
    registry.register(**FORECAST_TOOL)

    # Analysis
    registry.register(**DESCRIBE_TOOL)
    registry.register(**DECOMPOSE_TOOL)
    registry.register(**COMPARE_TOOL)
    registry.register(**COMPARE_PERIODS_TOOL)

    # Anomaly detection
    registry.register(**ANOMALY_TOOL)

    # Simulation
    registry.register(**WHAT_IF_TOOL)
    registry.register(**SENSITIVITY_TOOL)

    # Backtesting
    registry.register(**BACKTEST_TOOL)

    return registry


# Legacy TOOLS dict for backwards compatibility
TOOLS = {
    "run_forecast": {
        "function": run_forecast,
        "description": FORECAST_TOOL["description"],
        "parameters": FORECAST_TOOL["parameters"]
    },
    "describe_series": {
        "function": describe_series,
        "description": DESCRIBE_TOOL["description"],
        "parameters": DESCRIBE_TOOL["parameters"]
    },
    "decompose_series": {
        "function": decompose_series,
        "description": DECOMPOSE_TOOL["description"],
        "parameters": DECOMPOSE_TOOL["parameters"]
    },
    "compare_series": {
        "function": compare_series,
        "description": COMPARE_TOOL["description"],
        "parameters": COMPARE_TOOL["parameters"]
    },
    "detect_anomalies": {
        "function": detect_anomalies,
        "description": ANOMALY_TOOL["description"],
        "parameters": ANOMALY_TOOL["parameters"]
    },
    "what_if_simulation": {
        "function": what_if_simulation,
        "description": WHAT_IF_TOOL["description"],
        "parameters": WHAT_IF_TOOL["parameters"]
    },
    "sensitivity_analysis": {
        "function": sensitivity_analysis,
        "description": SENSITIVITY_TOOL["description"],
        "parameters": SENSITIVITY_TOOL["parameters"]
    },
    "backtest_forecast": {
        "function": backtest_forecast,
        "description": BACKTEST_TOOL["description"],
        "parameters": BACKTEST_TOOL["parameters"]
    },
    "compare_periods": {
        "function": compare_periods,
        "description": COMPARE_PERIODS_TOOL["description"],
        "parameters": COMPARE_PERIODS_TOOL["parameters"]
    }
}


__all__ = [
    # Registry
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "get_tools",
    "register_all_tools",
    "TOOLS",
    # Tool functions
    "run_forecast",
    "describe_series",
    "decompose_series",
    "compare_series",
    "detect_anomalies",
    "what_if_simulation",
    "sensitivity_analysis",
    "backtest_forecast",
    "compare_periods",
]
