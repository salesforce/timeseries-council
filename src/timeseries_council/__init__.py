# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
timeseries-council: AI Council for Time Series Analysis

A multi-model forecasting and anomaly detection library with LLM deliberation.

Usage with CSV file:
    from timeseries_council import Orchestrator, Config

    config = Config()
    orchestrator = Orchestrator(
        llm_provider=config.get_provider("anthropic"),
        csv_path="data/sales.csv",
        target_col="sales"
    )
    response = orchestrator.chat("What will sales be next week?")

Usage with pd.Series (for library / programmatic use):
    import pandas as pd
    from timeseries_council import Orchestrator, Config
    from timeseries_council.tools import detect_anomalies, run_forecast

    # Direct tool usage (no LLM required)
    series = pd.Series(
        [10.0, 11.2, 10.8, 50.0, 10.9],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    result = detect_anomalies(series=series)
    result = run_forecast(series=series, horizon=3)

    # With Orchestrator
    orchestrator = Orchestrator(
        llm_provider=config.get_provider("anthropic"),
        data=series,
    )
    response = orchestrator.chat("Are there any anomalies?")
"""

from .version import __version__, __version_info__
from .logging import configure_logging, get_logger, set_level
from .exceptions import (
    TimeseriesCouncilError,
    ConfigurationError,
    ProviderError,
    ForecasterError,
    DetectorError,
    ToolError,
    DataError,
    CouncilError,
)
from .types import (
    ChatMode,
    AnomalyType,
    ProgressStage,
    ForecastResult,
    DetectionResult,
    DetectionMemory,
    Anomaly,
    ToolCall,
    CouncilPerspective,
)
from .config import Config
from .orchestrator import Orchestrator

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Logging
    "configure_logging",
    "get_logger",
    "set_level",
    # Exceptions
    "TimeseriesCouncilError",
    "ConfigurationError",
    "ProviderError",
    "ForecasterError",
    "DetectorError",
    "ToolError",
    "DataError",
    "CouncilError",
    # Types
    "ChatMode",
    "AnomalyType",
    "ProgressStage",
    "ForecastResult",
    "DetectionResult",
    "DetectionMemory",
    "Anomaly",
    "ToolCall",
    "CouncilPerspective",
    # Main classes
    "Config",
    "Orchestrator",
]
