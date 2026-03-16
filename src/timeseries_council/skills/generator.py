# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Dynamic Skill Generator - Generate and execute skills at runtime using LLM.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import re

from .registry import Skill, SkillParameter, SkillRegistry, get_registry
from .sandbox import CodeSandbox
from .executor import SkillResult, DataContext
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str
    function_name: str
    description: str
    reasoning: str
    imports: List[str] = field(default_factory=list)


class DynamicSkillGenerator:
    """
    Generate and execute skills dynamically using LLM.

    When a user query doesn't match any existing skill, this generator:
    1. Uses the LLM to generate appropriate Python code
    2. Validates and sandboxes the code
    3. Executes it safely
    4. Optionally caches as a new skill
    """

    CODE_GENERATION_PROMPT = """You are a Python code generator for time series analysis.

Given a user query and data context, generate Python code to solve the problem.

CONSTRAINTS:
- Only use: numpy (as np), pandas (as pd), scipy, sklearn, math, statistics
- The data is provided as a pandas DataFrame called `data`
- The target column name is in `target_col`
- Return results in a variable called `result` (dict format preferred)
- Code must be self-contained and executable
- No file I/O, network calls, or system commands
- Maximum 50 lines of code

QUERY: {query}

DATA INFO:
- Shape: {shape}
- Columns: {columns}
- Target: {target_col}
- Sample: {sample}

USER CONTEXT: {user_context}

Generate Python code that addresses this query.

Respond in JSON format:
```json
{{
    "reasoning": "Brief explanation of the approach",
    "code": "Python code here",
    "function_name": "analyze_data",
    "description": "What this code does"
}}
```"""

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        registry: Optional[SkillRegistry] = None,
        sandbox: Optional[CodeSandbox] = None,
        cache_generated: bool = True,
    ):
        """
        Initialize the dynamic skill generator.

        Args:
            llm_provider: LLM provider for code generation
            registry: Skill registry for caching generated skills
            sandbox: Code sandbox for safe execution
            cache_generated: Whether to cache generated skills
        """
        self.llm_provider = llm_provider
        self.registry = registry or get_registry()
        self.sandbox = sandbox or CodeSandbox()
        self.cache_generated = cache_generated

    def can_generate(self) -> bool:
        """Check if generation is possible (LLM available)."""
        return self.llm_provider is not None

    async def generate_code(
        self,
        query: str,
        data_context: DataContext,
    ) -> Optional[GeneratedCode]:
        """
        Generate code using the LLM.

        Args:
            query: User's query
            data_context: Data context

        Returns:
            GeneratedCode object or None on failure
        """
        if not self.llm_provider:
            logger.warning("No LLM provider available for code generation")
            return None

        # Build the prompt
        data = data_context.data
        prompt = self.CODE_GENERATION_PROMPT.format(
            query=query,
            shape=str(data.shape) if data is not None else "No data",
            columns=list(data.columns) if data is not None else [],
            target_col=data_context.target_col,
            sample=data.head(3).to_dict() if data is not None else {},
            user_context=data_context.user_context or "None provided",
        )

        try:
            # Call LLM
            response = await self.llm_provider.generate(prompt)

            # Parse response
            generated = self._parse_llm_response(response)

            if generated:
                logger.info(f"Generated code for query: {query[:50]}...")
                return generated

        except Exception as e:
            logger.error(f"Error generating code: {e}")

        return None

    def generate_code_sync(
        self,
        query: str,
        data_context: DataContext,
    ) -> Optional[GeneratedCode]:
        """
        Synchronous version of generate_code.

        Args:
            query: User's query
            data_context: Data context

        Returns:
            GeneratedCode object or None on failure
        """
        if not self.llm_provider:
            logger.warning("No LLM provider available for code generation")
            return None

        # Build the prompt
        data = data_context.data
        prompt = self.CODE_GENERATION_PROMPT.format(
            query=query,
            shape=str(data.shape) if data is not None else "No data",
            columns=list(data.columns) if data is not None else [],
            target_col=data_context.target_col,
            sample=data.head(3).to_dict() if data is not None else {},
            user_context=data_context.user_context or "None provided",
        )

        try:
            # Call LLM (sync)
            if hasattr(self.llm_provider, 'generate_sync'):
                response = self.llm_provider.generate_sync(prompt)
            elif hasattr(self.llm_provider, 'generate'):
                # Try calling generate directly (might be sync)
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Already in async context, can't use run_until_complete
                        logger.warning("Cannot generate code synchronously in async context")
                        return None
                    response = loop.run_until_complete(self.llm_provider.generate(prompt))
                except RuntimeError:
                    response = asyncio.run(self.llm_provider.generate(prompt))
            else:
                logger.error("LLM provider has no generate method")
                return None

            # Parse response
            generated = self._parse_llm_response(response)

            if generated:
                logger.info(f"Generated code for query: {query[:50]}...")
                return generated

        except Exception as e:
            logger.error(f"Error generating code: {e}")

        return None

    def _parse_llm_response(self, response: str) -> Optional[GeneratedCode]:
        """Parse the LLM response into GeneratedCode."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try parsing entire response as JSON
                json_str = response

            data = json.loads(json_str)

            return GeneratedCode(
                code=data.get("code", ""),
                function_name=data.get("function_name", "analyze_data"),
                description=data.get("description", ""),
                reasoning=data.get("reasoning", ""),
            )

        except json.JSONDecodeError:
            # Try to extract code directly
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                return GeneratedCode(
                    code=code_match.group(1),
                    function_name="analyze_data",
                    description="Generated analysis",
                    reasoning="",
                )

        return None

    def execute_generated(
        self,
        generated: GeneratedCode,
        data_context: DataContext,
    ) -> SkillResult:
        """
        Execute generated code in the sandbox.

        Args:
            generated: Generated code object
            data_context: Data context

        Returns:
            SkillResult from execution
        """
        # Validate code
        is_valid, error = self.sandbox.validate_code(generated.code)

        if not is_valid:
            return SkillResult(
                skill_name="generated",
                success=False,
                error=f"Generated code validation failed: {error}",
                thinking=generated.reasoning,
            )

        # Prepare inputs
        inputs = {
            "data": data_context.data,
            "target_col": data_context.target_col,
        }

        if data_context.user_context:
            inputs["user_context"] = data_context.user_context

        # Execute
        result = self.sandbox.execute(generated.code, inputs)

        if result["success"]:
            return SkillResult(
                skill_name="generated",
                success=True,
                data=self._serialize_result(result["result"]),
                thinking=generated.reasoning,
                metadata={
                    "generated_code": generated.code,
                    "description": generated.description,
                    "stdout": result["output"],
                },
            )
        else:
            return SkillResult(
                skill_name="generated",
                success=False,
                error=result["error"],
                thinking=generated.reasoning,
                metadata={
                    "generated_code": generated.code,
                },
            )

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Serialize execution result to JSON-safe format."""
        if result is None:
            return {}

        if isinstance(result, dict):
            return {k: self._serialize_value(v) for k, v in result.items()}

        return {"value": self._serialize_value(result)}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        import numpy as np
        import pandas as pd

        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, pd.DataFrame):
            return value.to_dict()
        elif isinstance(value, pd.Series):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    def generate_and_execute(
        self,
        query: str,
        data_context: DataContext,
    ) -> SkillResult:
        """
        Generate code for a query and execute it.

        Args:
            query: User's query
            data_context: Data context

        Returns:
            SkillResult from execution
        """
        # Generate code
        generated = self.generate_code_sync(query, data_context)

        if not generated:
            return SkillResult(
                skill_name="generated",
                success=False,
                error="Failed to generate code for this query",
            )

        # Execute
        result = self.execute_generated(generated, data_context)

        # Cache as skill if successful and caching enabled
        if result.success and self.cache_generated:
            self._cache_as_skill(query, generated)

        return result

    def _cache_as_skill(self, query: str, generated: GeneratedCode) -> None:
        """Cache generated code as a reusable skill."""
        # Create a simple name from the query
        name = self._create_skill_name(query)

        # Create skill
        skill = Skill(
            name=name,
            description=generated.description,
            triggers=self._extract_triggers(query),
            parameters=[],
            requires_data=True,
            multi_model=False,
            content=generated.code,
            is_generated=True,
        )

        # Register
        self.registry.register_generated(skill)
        logger.info(f"Cached generated skill: {name}")

    def _create_skill_name(self, query: str) -> str:
        """Create a skill name from query."""
        # Simple approach: take first few words
        words = query.lower().split()[:3]
        name = "_".join(w for w in words if w.isalnum())
        return f"gen_{name}"

    def _extract_triggers(self, query: str) -> List[str]:
        """Extract trigger words from query."""
        # Simple keyword extraction
        common_words = {"the", "a", "an", "is", "are", "what", "how", "can", "do", "this"}
        words = query.lower().split()
        triggers = [w for w in words if w.isalnum() and w not in common_words]
        return triggers[:5]
