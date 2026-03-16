# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Chat history management for sessions.

Stores conversation history in temp files to enable follow-up questions.
Implements smart truncation to avoid sending large data arrays to LLM.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile


# Directory for storing chat histories
CHAT_HISTORY_DIR = Path(tempfile.gettempdir()) / "timeseries_council_sessions"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class ChatHistoryManager:
    """
    Manages chat history storage in temp files.
    
    Stores full messages for UI display but provides truncated versions
    for LLM context to avoid token bloat from large prediction arrays.
    """
    
    def __init__(self, session_id: str, max_messages: int = 100):
        """
        Initialize chat history manager.
        
        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to keep (oldest removed first)
        """
        self.session_id = session_id
        self.max_messages = max_messages
        self.history_file = CHAT_HISTORY_DIR / f"{session_id}_chat.json"
    
    def save_message(
        self, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a message to chat history.
        
        Args:
            role: 'user' or 'assistant'
            content: The message text
            metadata: Optional metadata (skill_result, thinking, etc.)
        """
        # Load existing history
        history = self._load_history()
        
        # Create message entry
        message = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "metadata": self._truncate_metadata(metadata) if metadata else {}
        }
        
        # Append and enforce max limit
        history["messages"].append(message)
        if len(history["messages"]) > self.max_messages:
            history["messages"] = history["messages"][-self.max_messages:]
        
        # Save back to file
        self._save_history(history)
    
    def get_all_messages(self) -> List[Dict[str, Any]]:
        """
        Retrieve all messages for this session.
        
        Returns:
            List of message dictionaries with full metadata
        """
        history = self._load_history()
        return history.get("messages", [])
    
    def get_recent_context(self, last_n: int = 3) -> List[Dict[str, str]]:
        """
        Get last N message pairs for LLM context (truncated).
        
        This returns ONLY the conversational text, excluding:
        - Large data arrays (predictions, values, anomalies)
        - Full tool results
        - Chart data
        
        Args:
            last_n: Number of recent message pairs to include
            
        Returns:
            List of {role, content} dicts suitable for LLM context
        """
        all_messages = self.get_all_messages()
        
        # Get last N*2 messages (N user + N assistant pairs)
        recent = all_messages[-(last_n * 2):]
        
        # Extract only role and content, with smart summarization
        context = []
        for msg in recent:
            context_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # For assistant messages, add brief metadata summary
            if msg["role"] == "assistant" and msg.get("metadata"):
                meta = msg["metadata"]
                summary_parts = []
                
                # Add skill name if available
                if meta.get("skill_name"):
                    summary_parts.append(f"[Used: {meta['skill_name']}]")
                
                # Add models used if available
                if meta.get("models_used"):
                    models = meta["models_used"]
                    if isinstance(models, list) and models:
                        summary_parts.append(f"[Models: {', '.join(models)}]")
                
                # Add data summary if available (NOT the full data)
                if meta.get("data_summary"):
                    summary_parts.append(meta["data_summary"])
                
                if summary_parts:
                    context_msg["content"] = " ".join(summary_parts) + "\n\n" + context_msg["content"]
            
            context.append(context_msg)
        
        return context
    
    def clear_history(self) -> None:
        """Clear all chat history for this session."""
        if self.history_file.exists():
            self.history_file.unlink()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load history from file or return empty structure."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted file, start fresh
                return {"messages": [], "session_id": self.session_id}
        else:
            return {"messages": [], "session_id": self.session_id}
    
    def _save_history(self, history: Dict[str, Any]) -> None:
        """Save history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Log error but don't crash - chat can continue without history
            print(f"Warning: Failed to save chat history: {e}")
    
    def _truncate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Truncate metadata to avoid storing huge arrays.
        
        Keeps:
        - skill_name, models_used, execution_time
        - Brief summaries of results
        
        Removes:
        - Large arrays (predictions, values, anomalies)
        - Full tool results with data
        - Raw chart data
        """
        if not metadata:
            return {}
        
        truncated = {}
        
        # Keep these fields as-is (small metadata)
        for key in ["skill_name", "models_used", "execution_time", "success"]:
            if key in metadata:
                truncated[key] = metadata[key]
        
        # Create a data summary instead of storing full data
        if "data" in metadata or "skill_result" in metadata:
            data = metadata.get("data") or metadata.get("skill_result", {})
            truncated["data_summary"] = self._summarize_data(data)
        
        # Keep thinking but truncate if too long
        if "thinking" in metadata:
            thinking = metadata["thinking"]
            if isinstance(thinking, str) and len(thinking) > 500:
                truncated["thinking"] = thinking[:500] + "..."
            elif isinstance(thinking, dict):
                # Just keep skill_selection
                truncated["thinking"] = thinking.get("skill_selection", "")
            else:
                truncated["thinking"] = thinking
        
        return truncated
    
    def _summarize_data(self, data: Any) -> str:
        """
        Create a brief text summary of data results.
        
        Examples:
        - "Forecasted 14 points"
        - "Detected 3 anomalies"
        - "Analyzed trend over 100 points"
        """
        if not data or not isinstance(data, dict):
            return ""
        
        summary_parts = []
        
        # Forecast data
        if "predictions" in data or "forecast" in data:
            pred_data = data.get("predictions") or data.get("forecast")
            if isinstance(pred_data, list):
                summary_parts.append(f"Forecasted {len(pred_data)} points")
            elif isinstance(pred_data, dict) and "mean_prediction" in pred_data:
                mean_pred = pred_data["mean_prediction"]
                if isinstance(mean_pred, list):
                    summary_parts.append(f"Forecasted {len(mean_pred)} points")
        
        # Anomaly data
        anomaly_count = 0
        for key in ["anomaly_indices", "anomalies", "high_confidence_anomalies"]:
            if key in data and isinstance(data[key], list):
                anomaly_count = max(anomaly_count, len(data[key]))
        if anomaly_count > 0:
            summary_parts.append(f"Detected {anomaly_count} anomalies")
        
        # Decomposition
        if "trend" in data and isinstance(data["trend"], list):
            summary_parts.append(f"Decomposed {len(data['trend'])} points")
        
        # Multi-model results
        if "model_results" in data and isinstance(data["model_results"], dict):
            model_count = len(data["model_results"])
            summary_parts.append(f"Compared {model_count} models")
        
        # Default summary
        if not summary_parts and isinstance(data, dict):
            if "success" in data:
                summary_parts.append("Analysis completed")
        
        return "; ".join(summary_parts) if summary_parts else ""


def cleanup_old_histories(max_age_hours: int = 24) -> int:
    """
    Clean up chat histories older than specified hours.
    
    Args:
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of files cleaned up
    """
    if not CHAT_HISTORY_DIR.exists():
        return 0
    
    import time
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    cleaned = 0
    
    for history_file in CHAT_HISTORY_DIR.glob("*_chat.json"):
        try:
            file_age = current_time - history_file.stat().st_mtime
            if file_age > max_age_seconds:
                history_file.unlink()
                cleaned += 1
        except (OSError, IOError):
            # Skip files we can't access
            continue
    
    return cleaned
