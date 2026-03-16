# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Custom exception hierarchy for timeseries-council.

Exception tree:
TimeseriesCouncilError (base)
|-- ConfigurationError
|   |-- ProviderConfigError
|   |-- MissingAPIKeyError
|-- ProviderError
|   |-- ProviderAuthError
|   |-- ProviderRateLimitError
|   |-- ProviderUnavailableError
|-- ForecasterError
|   |-- ModelNotFoundError
|   |-- InsufficientDataError
|   |-- ForecastExecutionError
|-- DetectorError
|   |-- DetectorFitError
|   |-- DetectionExecutionError
|-- ToolError
|   |-- InvalidToolError
|   |-- ToolExecutionError
|-- DataError
|   |-- FileNotFoundError
|   |-- InvalidColumnError
|   |-- DataParsingError
|-- CouncilError
|   |-- CouncilTimeoutError
"""

from typing import Any, Dict, List, Optional


class TimeseriesCouncilError(Exception):
    """Base exception for all timeseries-council errors."""

    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None, **kwargs):
        # accept and preserve extra keyword args (e.g., provider) to avoid unexpected-kwarg errors
        self.message = message or ""
        merged = dict(details or {})
        if kwargs:
            merged.update(kwargs)
        self.details = merged
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(TimeseriesCouncilError):
    """Error in configuration."""
    pass


class ProviderConfigError(ConfigurationError):
    """Error in provider configuration."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""

    def __init__(self, provider: str, env_var: str):
        super().__init__(
            f"API key for '{provider}' not found. Set {env_var} environment variable or add 'api_key' to config.yaml",
            {"provider": provider, "env_var": env_var}
        )
        self.provider = provider
        self.env_var = env_var


# ============================================================================
# Provider Errors
# ============================================================================

class ProviderError(TimeseriesCouncilError):
    """Error from LLM provider."""
    pass


class ProviderAuthError(ProviderError):
    """Authentication failed with provider."""

    def __init__(self, provider: str, message: str = "Authentication failed"):
        super().__init__(
            f"{provider}: {message}",
            {"provider": provider}
        )
        self.provider = provider


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, provider: str, retry_after: Optional[int] = None):
        msg = f"Rate limit exceeded for {provider}"
        if retry_after:
            msg += f". Retry after {retry_after} seconds"
        super().__init__(msg, {"provider": provider, "retry_after": retry_after})
        self.provider = provider
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """Provider service is unavailable."""

    def __init__(self, provider: str, reason: str = "Service unavailable"):
        super().__init__(
            f"{provider}: {reason}",
            {"provider": provider}
        )
        self.provider = provider


# ============================================================================
# Forecaster Errors
# ============================================================================

class ForecasterError(TimeseriesCouncilError):
    """Error in forecasting."""
    pass


class ModelNotFoundError(ForecasterError):
    """Requested forecasting model not found."""

    def __init__(self, model_name: str, available_models: Optional[List[str]] = None):
        super().__init__(
            f"Forecasting model '{model_name}' not found",
            {"model_name": model_name, "available_models": available_models or []}
        )
        self.model_name = model_name
        self.available_models = available_models or []


class InsufficientDataError(ForecasterError):
    """Not enough data for forecasting."""

    def __init__(self, required: int, provided: int, model: Optional[str] = None):
        msg = f"Insufficient data: need at least {required} points, got {provided}"
        if model:
            msg = f"{model}: {msg}"
        super().__init__(msg, {"required": required, "provided": provided, "model": model})
        self.required = required
        self.provided = provided
        self.model = model


class ForecastExecutionError(ForecasterError):
    """Error during forecast execution."""

    def __init__(self, model: str, reason: str):
        super().__init__(
            f"Forecast failed for {model}: {reason}",
            {"model": model, "reason": reason}
        )
        self.model = model
        self.reason = reason


# ============================================================================
# Detector Errors
# ============================================================================

class DetectorError(TimeseriesCouncilError):
    """Error in anomaly detection."""
    pass


class DetectorNotFoundError(DetectorError):
    """Requested detector not found."""

    def __init__(self, detector_name: str, available_detectors: Optional[List[str]] = None):
        super().__init__(
            f"Anomaly detector '{detector_name}' not found",
            {"detector_name": detector_name, "available_detectors": available_detectors or []}
        )
        self.detector_name = detector_name
        self.available_detectors = available_detectors or []


class DetectorFitError(DetectorError):
    """Error during detector fitting."""

    def __init__(self, detector: str, reason: str):
        super().__init__(
            f"Failed to fit detector {detector}: {reason}",
            {"detector": detector, "reason": reason}
        )
        self.detector = detector
        self.reason = reason


class DetectionExecutionError(DetectorError):
    """Error during anomaly detection."""

    def __init__(self, detector: str, reason: str):
        super().__init__(
            f"Detection failed for {detector}: {reason}",
            {"detector": detector, "reason": reason}
        )
        self.detector = detector
        self.reason = reason


# ============================================================================
# Tool Errors
# ============================================================================

class ToolError(TimeseriesCouncilError):
    """Error in tool execution."""
    pass


class InvalidToolError(ToolError):
    """Unknown tool requested."""

    def __init__(self, tool_name: str, available_tools: Optional[List[str]] = None):
        super().__init__(
            f"Unknown tool: {tool_name}",
            {"tool_name": tool_name, "available_tools": available_tools or []}
        )
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class ToolExecutionError(ToolError):
    """Error during tool execution."""

    def __init__(self, tool: str, reason: str):
        super().__init__(
            f"Tool '{tool}' execution failed: {reason}",
            {"tool": tool, "reason": reason}
        )
        self.tool = tool
        self.reason = reason


# ============================================================================
# Data Errors
# ============================================================================

class DataError(TimeseriesCouncilError):
    """Error with data processing."""
    pass


class DataFileNotFoundError(DataError):
    """Data file not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Data file not found: {path}",
            {"path": path}
        )
        self.path = path


class InvalidColumnError(DataError):
    """Column not found in dataset."""

    def __init__(self, column: str, available_columns: Optional[List[str]] = None):
        super().__init__(
            f"Column '{column}' not found in dataset",
            {"column": column, "available_columns": available_columns or []}
        )
        self.column = column
        self.available_columns = available_columns or []


class DataParsingError(DataError):
    """Error parsing data file."""

    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Failed to parse {path}: {reason}",
            {"path": path, "reason": reason}
        )
        self.path = path
        self.reason = reason


# ============================================================================
# Council Errors
# ============================================================================

class CouncilError(TimeseriesCouncilError):
    """Error in council deliberation."""
    pass


class CouncilTimeoutError(CouncilError):
    """Council deliberation timed out."""

    def __init__(self, stage: str, timeout: int):
        super().__init__(
            f"Council timed out during {stage} after {timeout}s",
            {"stage": stage, "timeout": timeout}
        )
        self.stage = stage
        self.timeout = timeout


# ============================================================================
# Orchestrator Errors
# ============================================================================

class OrchestratorError(TimeseriesCouncilError):
    """Error in orchestrator execution."""
    pass
