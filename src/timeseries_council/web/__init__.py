# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Web interface for timeseries-council.

Provides FastAPI-based web UI with:
- Real-time progress tracking via SSE
- Model selection for forecasters and detectors
- Standard, Council, and Advanced Council chat modes
"""

from .app import create_app, run_server, get_templates
from .progress import ProgressTracker, get_tracker
from .models import (
    ChatMode,
    SessionConfig,
    SessionResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProgressUpdate,
)

__all__ = [
    "create_app",
    "run_server",
    "get_templates",
    "ProgressTracker",
    "get_tracker",
    "ChatMode",
    "SessionConfig",
    "SessionResponse",
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "ProgressUpdate",
]
