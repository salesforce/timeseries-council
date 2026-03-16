# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Tool registry for the orchestrator.
"""

from typing import Dict, Any, Callable, List, Optional
from ..logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, str]
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name (used in LLM responses)
            function: The tool function
            description: Description for the LLM
            parameters: Parameter descriptions
        """
        self._tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }
        logger.debug(f"Registered tool: {name}")

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Tool arguments

        Returns:
            Tool result dict
        """
        tool = self._tools.get(name)
        if not tool:
            logger.error(f"Unknown tool: {name}")
            return {"success": False, "error": f"Unknown tool: {name}"}

        try:
            logger.info(f"Executing tool: {name}")
            result = tool["function"](**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {"success": False, "error": str(e)}

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_descriptions(self) -> Dict[str, str]:
        """Get tool descriptions for LLM context."""
        return {
            name: info["description"]
            for name, info in self._tools.items()
        }

    def get_full_info(self) -> Dict[str, Dict[str, Any]]:
        """Get full tool information."""
        return {
            name: {
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self._tools.items()
        }

    def format_for_prompt(self) -> str:
        """Format tool info for LLM system prompt."""
        lines = ["Available tools:"]
        for name, info in self._tools.items():
            lines.append(f"\n- {name}: {info['description']}")
            for param, desc in info["parameters"].items():
                lines.append(f"    {param}: {desc}")
        return "\n".join(lines)


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    function: Callable,
    description: str,
    parameters: Dict[str, str]
) -> None:
    """Register a tool in the global registry."""
    get_registry().register(name, function, description, parameters)


def get_tools() -> Dict[str, Dict[str, Any]]:
    """Get all tools as a dict (backwards compatibility)."""
    registry = get_registry()
    return {
        name: registry._tools[name]
        for name in registry.list_tools()
    }
