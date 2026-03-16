# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Council module for multi-perspective AI analysis.

Provides:
- MultiLLMCouncil: True Karpathy-style multi-LLM council (uses different providers)
- AdvancedCouncil: Single-LLM multi-role council (legacy)
- Council roles for different analysis perspectives
"""

from .council import AdvancedCouncil, run_full_council
from .roles import COUNCIL_ROLES, get_role_prompt, get_all_roles
from .multi_llm_council import MultiLLMCouncil, CouncilDeliberation

__all__ = [
    "MultiLLMCouncil",
    "CouncilDeliberation",
    "AdvancedCouncil",
    "run_full_council",
    "COUNCIL_ROLES",
    "get_role_prompt",
    "get_all_roles",
]
