# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Type definitions and protocols for timeseries-council.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict


# ============================================================================
# Enums
# ============================================================================

class ChatMode(str, Enum):
    """Chat mode options."""
    STANDARD = "standard"
    COUNCIL = "council"
    ADVANCED_COUNCIL = "advanced_council"


class AnomalyType(str, Enum):
    """Types of anomalies."""
    SPIKE = "spike"
    DROP = "drop"
    LEVEL_SHIFT = "level_shift"
    TREND_CHANGE = "trend_change"
    UNKNOWN = "unknown"


class ProgressStage(str, Enum):
    """Stages of operation for progress tracking."""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    COUNCIL = "council"  # General council processing
    COUNCIL_STAGE_1 = "council_stage_1"  # Expert analyses
    COUNCIL_STAGE_2 = "council_stage_2"  # Round-table discussion
    COUNCIL_STAGE_3 = "council_stage_3"  # Final synthesis
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


# ============================================================================
# Protocols (Structural Typing)
# ============================================================================

class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        progress: float,
        message: str,
        stage: Optional[ProgressStage] = None
    ) -> None:
        """
        Report progress.

        Args:
            progress: Progress from 0.0 to 1.0
            message: Human-readable progress message
            stage: Optional stage identifier
        """
        ...


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ForecastResult:
    """Result from a forecast operation."""
    success: bool
    forecast: Optional[List[float]] = None
    timestamps: Optional[List[str]] = None
    uncertainty: Optional[List[float]] = None
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    model_name: str = ""
    horizon: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantileForecastResult(ForecastResult):
    """Extended ForecastResult with full quantile predictions."""
    quantiles: Dict[str, List[float]] = field(default_factory=dict)
    # Keys are quantile levels as strings: "0.1", "0.2", ..., "0.9"
    # Values are lists of float predictions at that quantile level


@dataclass
class Anomaly:
    """Single detected anomaly."""
    timestamp: str
    value: float
    score: float
    anomaly_type: AnomalyType = AnomalyType.UNKNOWN
    confidence: Optional[float] = None
    explanation: Optional[str] = None


@dataclass
class DetectionMemory:
    """Contextual memory passed to detectors for stateful detection.

    Allows callers to provide historical context so detectors can make
    more informed decisions — baseline-aware scoring, expected-range
    filtering, and LLM-level textual reasoning.

    Attributes:
        baseline_stats: Known normal statistics from a reference period.
            Used by statistical detectors (Z-Score, MAD) to compute scores
            against the baseline instead of the current batch.
            e.g. {"mean": 100.0, "std": 10.0, "median": 98.0}
        expected_range: Acceptable value range [low, high]. Anomalies whose
            values fall inside this range are filtered out.
            e.g. [80.0, 120.0]
        context: Free-form textual or structured context for LLM-based
            detectors and post-processing. Can be a string description or
            a dict with arbitrary domain knowledge.
            e.g. "Holiday period, expect 20% higher sales"
            e.g. {"seasonality": "weekly", "notes": "promotion running"}
    """
    baseline_stats: Dict[str, float] = field(default_factory=dict)
    expected_range: Optional[List[float]] = None
    context: Any = None


@dataclass
class DetectionResult:
    """Result from anomaly detection."""
    success: bool
    anomalies: List[Anomaly] = field(default_factory=list)
    anomaly_count: int = 0
    detector_name: str = ""
    threshold: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.anomaly_count == 0 and self.anomalies:
            self.anomaly_count = len(self.anomalies)


@dataclass
class ProgressUpdate:
    """Progress update message."""
    stage: ProgressStage
    progress: float
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Parsed tool call from LLM."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CouncilPerspective:
    """A single council perspective."""
    role: str
    analysis: str
    provider: Optional[str] = None


# ============================================================================
# TypedDicts (for dictionary type hints)
# ============================================================================

class ToolResultDict(TypedDict, total=False):
    """Standard tool result format."""
    success: bool
    error: Optional[str]


class ForecastResultDict(TypedDict, total=False):
    """Forecast tool result format."""
    success: bool
    forecast: List[float]
    timestamps: List[str]
    uncertainty: Optional[List[float]]
    horizon: int
    model: str
    error: Optional[str]


class AnomalyDict(TypedDict, total=False):
    """Single anomaly in dictionary form."""
    timestamp: str
    value: float
    score: float
    type: str
    confidence: Optional[float]
    explanation: Optional[str]


class DetectionResultDict(TypedDict, total=False):
    """Anomaly detection result format."""
    success: bool
    anomaly_count: int
    anomalies: List[AnomalyDict]
    sensitivity: float
    detector: str
    error: Optional[str]


class SessionConfigDict(TypedDict, total=False):
    """Session configuration."""
    csv_path: str
    target_col: str
    provider: Optional[str]
    forecaster: Optional[str]
    detector: Optional[str]


class ChatResponseDict(TypedDict, total=False):
    """Chat response format."""
    response: str
    tool_call: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    mode: str
    council_perspectives: Optional[List[Dict[str, Any]]]
    advanced_council: Optional[Dict[str, Any]]
