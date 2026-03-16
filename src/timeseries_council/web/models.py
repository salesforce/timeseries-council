# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Pydantic models for web API requests and responses.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel


class ChatMode(str, Enum):
    """Chat mode options."""
    STANDARD = "standard"
    COUNCIL = "council"
    ADVANCED_COUNCIL = "advanced_council"
    MULTI_MODEL = "multi_model"


class SessionConfig(BaseModel):
    """Session configuration request."""
    csv_path: str
    target_col: str = "sales"
    provider: str = "gemini"
    forecaster: str = "moirai"
    detector: str = "zscore"


class SessionResponse(BaseModel):
    """Session creation response."""
    session_id: str
    status: str
    provider: str
    csv_path: str
    target_col: str
    forecaster: str = "moirai"
    detector: str = "zscore"
    file_name: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call information."""
    tool: str
    args: Dict[str, Any]


class CouncilPerspective(BaseModel):
    """A single council perspective."""
    role: str
    analysis: str
    provider: Optional[str] = None


class AdvancedCouncilStage1(BaseModel):
    """Stage 1 result: initial response from a model."""
    model: str
    response: str
    provider: Optional[str] = None


class AdvancedCouncilStage2(BaseModel):
    """Stage 2 result: ranking from a model."""
    model: str
    ranking: str
    parsed_ranking: Optional[List[str]] = None
    provider: Optional[str] = None


class AdvancedCouncilStage3(BaseModel):
    """Stage 3 result: chairman synthesis."""
    model: str
    response: str
    is_fallback: bool = False


class AggregateRanking(BaseModel):
    """Aggregate ranking result."""
    model: str
    score: float
    average_rank: float = 0.0


class AdvancedCouncilResult(BaseModel):
    """Full advanced council result."""
    stage1: List[AdvancedCouncilStage1]
    stage2: List[AdvancedCouncilStage2]
    stage3: AdvancedCouncilStage3
    aggregate_rankings: Optional[List[AggregateRanking]] = None
    chairman: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat message request."""
    message: str
    mode: ChatMode = ChatMode.STANDARD
    forecaster: Optional[str] = None
    detector: Optional[str] = None
    user_context: Optional[str] = None


class SkillExecution(BaseModel):
    """Skill execution result."""
    skill_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    models_used: Optional[List[str]] = None
    execution_time: Optional[float] = None


class CouncilExpert(BaseModel):
    """A single council expert's analysis."""
    key: str
    name: str
    role: str
    emoji: str
    analysis: str


class CouncilDeliberation(BaseModel):
    """Full Karpathy-style council deliberation."""
    experts: List[CouncilExpert] = []
    round_table: List[Dict[str, Any]] = []
    synthesis: Optional[Dict[str, Any]] = None
    full_transcript: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response with optional tool and council info."""
    response: str
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[Dict[str, Any]] = None
    skill_result: Optional[SkillExecution] = None
    thinking: Optional[Any] = None  # Can be string or dict
    council_perspectives: Optional[List[CouncilPerspective]] = None
    perspectives: Optional[Dict[str, str]] = None
    deliberation: Optional[CouncilDeliberation] = None  # Karpathy-style deliberation
    advanced_council: Optional[AdvancedCouncilResult] = None
    mode: ChatMode = ChatMode.STANDARD
    suggestions: Optional[List[str]] = None


class ProgressUpdate(BaseModel):
    """Progress update for SSE streaming."""
    stage: str
    message: str
    progress: float  # 0.0 to 1.0
    complete: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    providers_available: List[str]
    forecasters_available: Optional[List[str]] = None
    detectors_available: Optional[List[str]] = None
