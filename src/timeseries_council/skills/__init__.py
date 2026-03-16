# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Skills architecture for timeseries-council.

Anthropic-style skills with SKILL.md definitions, multi-model support,
and dynamic skill generation.
"""

from .registry import SkillRegistry, Skill, SkillParameter, get_registry, reset_registry
from .loader import SkillLoader, load_skills
from .executor import SkillExecutor, SkillResult, DataContext
from .sandbox import CodeSandbox
from .generator import DynamicSkillGenerator, GeneratedCode

__all__ = [
    # Registry
    "SkillRegistry",
    "Skill",
    "SkillParameter",
    "get_registry",
    "reset_registry",
    # Loader
    "SkillLoader",
    "load_skills",
    # Executor
    "SkillExecutor",
    "SkillResult",
    "DataContext",
    # Sandbox
    "CodeSandbox",
    # Generator
    "DynamicSkillGenerator",
    "GeneratedCode",
]
