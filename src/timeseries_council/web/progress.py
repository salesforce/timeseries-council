# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Progress tracking with Server-Sent Events (SSE) support.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional, Callable
from dataclasses import dataclass, field
from queue import Queue
from threading import Lock

from ..types import ProgressStage
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressState:
    """Current progress state."""
    stage: ProgressStage = ProgressStage.INITIALIZING
    message: str = "Starting..."
    progress: float = 0.0
    complete: bool = False


class ProgressTracker:
    """Thread-safe progress tracker with SSE support."""

    def __init__(self):
        self._state = ProgressState()
        self._lock = Lock()
        self._queue: Queue = Queue()
        self._listeners: list = []

    @property
    def state(self) -> ProgressState:
        """Get current progress state."""
        with self._lock:
            return ProgressState(
                stage=self._state.stage,
                message=self._state.message,
                progress=self._state.progress,
                complete=self._state.complete
            )

    def update(self, stage: ProgressStage, message: str, progress: float):
        """Update progress state."""
        with self._lock:
            self._state.stage = stage
            self._state.message = message
            self._state.progress = min(1.0, max(0.0, progress))
            self._state.complete = stage == ProgressStage.COMPLETE

        # Notify listeners
        self._notify()
        logger.debug(f"Progress: {stage.value} - {progress:.0%} - {message}")

    def complete(self, message: str = "Complete"):
        """Mark as complete."""
        self.update(ProgressStage.COMPLETE, message, 1.0)

    def error(self, message: str):
        """Mark as error."""
        with self._lock:
            self._state.stage = ProgressStage.ERROR
            self._state.message = message
            self._state.progress = 0.0
            self._state.complete = True
        self._notify()

    def reset(self):
        """Reset progress state."""
        with self._lock:
            self._state = ProgressState()
        logger.debug("Progress tracker reset")

    def get_callback(self) -> Callable[[ProgressStage, str, float], None]:
        """Get a callback function for progress updates."""
        def callback(stage: ProgressStage, message: str, progress: float):
            self.update(stage, message, progress)
        return callback

    def _notify(self):
        """Notify all listeners of state change."""
        state = self.state
        event_data = {
            "stage": state.stage.value,
            "message": state.message,
            "progress": state.progress,
            "complete": state.complete
        }
        self._queue.put(event_data)

    async def stream_events(self) -> AsyncGenerator[str, None]:
        """
        Generate SSE events for progress updates.

        Yields:
            SSE-formatted event strings
        """
        while True:
            try:
                # Check for updates with timeout
                while not self._queue.empty():
                    event_data = self._queue.get_nowait()
                    yield f"data: {json.dumps(event_data)}\n\n"

                    if event_data.get("complete"):
                        return

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in SSE stream: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return


# Global tracker registry
_trackers: dict = {}
_tracker_lock = Lock()


def get_tracker(session_id: str) -> ProgressTracker:
    """Get or create a progress tracker for a session."""
    with _tracker_lock:
        if session_id not in _trackers:
            _trackers[session_id] = ProgressTracker()
        return _trackers[session_id]


def remove_tracker(session_id: str):
    """Remove a progress tracker."""
    with _tracker_lock:
        if session_id in _trackers:
            del _trackers[session_id]


def format_sse_event(data: dict, event: Optional[str] = None) -> str:
    """Format data as SSE event."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)
