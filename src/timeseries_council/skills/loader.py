# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Skill Loader - Parse SKILL.md files into Skill objects.
"""

import yaml
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from .registry import Skill, SkillParameter, SkillRegistry, get_registry
from ..logging import get_logger

logger = get_logger(__name__)


class SkillLoader:
    """
    Load skills from SKILL.md files.

    SKILL.md format:
    ```
    ---
    name: skill_name
    description: |
      Multi-line description of the skill.
      Use when: specific trigger conditions.
    triggers:
      - keyword1
      - keyword2
    parameters:
      - name: param1
        type: string
        default: "value"
    requires_data: true
    multi_model: false
    ---

    # Skill Name

    ## Instructions
    Detailed instructions for the skill...
    ```
    """

    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    def __init__(self, skills_dir: Optional[Path] = None):
        """
        Initialize the loader.

        Args:
            skills_dir: Directory containing SKILL.md files
        """
        if skills_dir is None:
            skills_dir = Path(__file__).parent / "definitions"
        self.skills_dir = skills_dir

    def load_all(self, registry: Optional[SkillRegistry] = None) -> List[Skill]:
        """
        Load all SKILL.md files from the skills directory.

        Args:
            registry: Optional registry to register skills into

        Returns:
            List of loaded Skill objects
        """
        if registry is None:
            registry = get_registry()

        skills = []

        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return skills

        # Load .md files
        for md_file in self.skills_dir.glob("*.md"):
            if md_file.name.lower() == "readme.md":
                continue

            try:
                skill = self.load_file(md_file)
                if skill:
                    skills.append(skill)
                    registry.register(skill)
            except Exception as e:
                logger.error(f"Failed to load skill from {md_file}: {e}")

        logger.info(f"Loaded {len(skills)} skills from {self.skills_dir}")
        return skills

    def load_file(self, file_path: Path) -> Optional[Skill]:
        """
        Load a single SKILL.md file.

        Args:
            file_path: Path to the SKILL.md file

        Returns:
            Skill object or None if invalid
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            return self.parse(content, file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def parse(self, content: str, file_path: Optional[Path] = None) -> Optional[Skill]:
        """
        Parse SKILL.md content into a Skill object.

        Args:
            content: Raw markdown content with YAML frontmatter
            file_path: Optional source file path

        Returns:
            Skill object or None if invalid
        """
        # Extract frontmatter
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            logger.warning(f"No valid frontmatter found in {file_path}")
            return None

        frontmatter_str = match.group(1)
        markdown_content = match.group(2)

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {file_path}: {e}")
            return None

        if not isinstance(frontmatter, dict):
            logger.error(f"Frontmatter must be a dict in {file_path}")
            return None

        # Required fields
        name = frontmatter.get('name')
        description = frontmatter.get('description', '')

        if not name:
            logger.error(f"Missing 'name' in {file_path}")
            return None

        # Parse triggers
        triggers = frontmatter.get('triggers', [])
        if isinstance(triggers, str):
            triggers = [triggers]

        # Parse parameters
        params_raw = frontmatter.get('parameters', [])
        parameters = []
        for p in params_raw:
            if isinstance(p, dict):
                parameters.append(SkillParameter(
                    name=p.get('name', ''),
                    type=p.get('type', 'string'),
                    description=p.get('description', ''),
                    default=p.get('default'),
                    required=p.get('required', False),
                    optional=p.get('optional', False),
                ))

        # Create skill
        skill = Skill(
            name=name,
            description=description,
            triggers=triggers,
            parameters=parameters,
            requires_data=frontmatter.get('requires_data', True),
            multi_model=frontmatter.get('multi_model', False),
            content=markdown_content.strip(),
            file_path=file_path,
        )

        logger.debug(f"Parsed skill: {name} with {len(parameters)} parameters")
        return skill

    def reload(self, registry: Optional[SkillRegistry] = None) -> List[Skill]:
        """
        Reload all skills (useful for development).

        Args:
            registry: Optional registry to reload into

        Returns:
            List of reloaded skills
        """
        if registry is None:
            registry = get_registry()

        # Clear existing skills (but not generated ones)
        registry._skills.clear()

        return self.load_all(registry)


def load_skills(skills_dir: Optional[Path] = None) -> List[Skill]:
    """
    Convenience function to load all skills.

    Args:
        skills_dir: Optional custom skills directory

    Returns:
        List of loaded skills
    """
    loader = SkillLoader(skills_dir)
    return loader.load_all()
