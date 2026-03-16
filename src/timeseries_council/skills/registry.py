# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Skill Registry - Core skill management and matching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import re

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SkillParameter:
    """Parameter definition for a skill."""
    name: str
    type: str  # 'string', 'integer', 'float', 'boolean', 'array'
    description: str = ""
    default: Any = None
    required: bool = False
    optional: bool = False


@dataclass
class Skill:
    """Skill definition loaded from SKILL.md."""
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    parameters: List[SkillParameter] = field(default_factory=list)
    requires_data: bool = True
    multi_model: bool = False
    content: str = ""  # Full markdown content after frontmatter
    file_path: Optional[Path] = None
    is_generated: bool = False

    # Runtime executor (set after loading)
    executor: Optional[Callable] = None

    def matches_query(self, query: str) -> float:
        """
        Check if this skill matches a natural language query.
        Returns a confidence score 0.0-1.0.
        """
        query_lower = query.lower()
        score = 0.0

        # Check triggers
        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                score = max(score, 0.8)
                # Exact word match gets higher score
                if re.search(rf'\b{re.escape(trigger.lower())}\b', query_lower):
                    score = max(score, 0.9)

        # Check description keywords
        desc_words = self.description.lower().split()
        query_words = set(query_lower.split())
        overlap = len(set(desc_words) & query_words)
        if overlap > 0:
            score = max(score, min(0.5 + overlap * 0.1, 0.7))

        return score

    def get_parameter(self, name: str) -> Optional[SkillParameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "default": p.default,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "requires_data": self.requires_data,
            "multi_model": self.multi_model,
            "is_generated": self.is_generated,
        }


class SkillRegistry:
    """
    Registry for managing skills.

    Loads skills from SKILL.md files and provides matching and execution.
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize the skill registry.

        Args:
            skills_dir: Directory containing SKILL.md files
        """
        self._skills: Dict[str, Skill] = {}
        self._generated_skills: Dict[str, Skill] = {}

        if skills_dir is None:
            skills_dir = Path(__file__).parent / "definitions"
        self.skills_dir = skills_dir

        logger.info(f"SkillRegistry initialized with dir: {skills_dir}")

    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill
        logger.info(f"Registered skill: {skill.name}")

    def register_generated(self, skill: Skill) -> None:
        """Register a dynamically generated skill."""
        skill.is_generated = True
        self._generated_skills[skill.name] = skill
        logger.info(f"Registered generated skill: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        # Check generated skills first (they may override built-in)
        if name in self._generated_skills:
            return self._generated_skills[name]
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """List all registered skill names."""
        all_skills = set(self._skills.keys()) | set(self._generated_skills.keys())
        return sorted(all_skills)

    def get_all(self) -> List[Skill]:
        """Get all registered skills."""
        skills = list(self._skills.values())
        skills.extend(self._generated_skills.values())
        return skills

    def match_query(self, query: str, threshold: float = 0.5) -> List[Skill]:
        """
        Find skills matching a natural language query.

        Args:
            query: Natural language query
            threshold: Minimum confidence score (0.0-1.0)

        Returns:
            List of matching skills, sorted by confidence
        """
        matches = []

        for skill in self.get_all():
            score = skill.matches_query(query)
            if score >= threshold:
                matches.append((skill, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return [skill for skill, score in matches]

    def get_for_prompt(self) -> str:
        """
        Get skill descriptions formatted for LLM prompt.

        Returns:
            Formatted string listing available skills
        """
        lines = ["Available Skills:"]

        for skill in self.get_all():
            triggers_str = ", ".join(skill.triggers[:3])
            lines.append(f"\n- **{skill.name}**: {skill.description[:100]}")
            lines.append(f"  Triggers: {triggers_str}")
            if skill.multi_model:
                lines.append("  Supports multi-model execution")

        return "\n".join(lines)

    def clear_generated(self) -> None:
        """Clear all generated skills."""
        self._generated_skills.clear()
        logger.info("Cleared all generated skills")


# Global registry instance
_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
