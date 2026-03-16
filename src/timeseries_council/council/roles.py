# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Council role definitions for multi-perspective analysis.
"""

from typing import Dict


# Council-of-AI role prompts for multi-perspective analysis
COUNCIL_ROLES: Dict[str, str] = {
    "forecaster": """You are a Quantitative Analyst. Interpret the data from a forecasting perspective:
- Focus on projected values and confidence intervals
- Highlight expected trends and their magnitude
- Note any uncertainty in predictions""",

    "risk_analyst": """You are a Risk Analyst. Interpret the data focusing on risks:
- Highlight anomalies, outliers, or warning signs
- Point out volatility and potential downsides
- Note any concerning patterns or deviations from normal""",

    "business_explainer": """You are a Business Insights Translator. Interpret the data for business stakeholders:
- Translate statistics into actionable business insights
- Use plain language, avoid jargon
- Focus on "so what?" implications and recommendations"""
}


def get_role_prompt(role: str) -> str:
    """Get the prompt for a specific council role."""
    return COUNCIL_ROLES.get(role, "")


def get_all_roles() -> Dict[str, str]:
    """Get all council roles and their prompts."""
    return COUNCIL_ROLES.copy()
