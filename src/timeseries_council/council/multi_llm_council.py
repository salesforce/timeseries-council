# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Karpathy-style Multi-LLM Council Implementation.

This implements the true LLM Council pattern from https://github.com/karpathy/llm-council

The council uses multiple different LLM providers (Claude, GPT, Gemini, etc.) with a 3-stage process:
1. Stage 1 - Independent Responses: Each LLM independently analyzes the question
2. Stage 2 - Peer Review: Each LLM ranks/judges other models' responses (anonymized)
3. Stage 3 - Chairman Synthesis: A designated chairman model synthesizes everything
"""

import os
import json
import random
import string
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..providers import create_provider, get_available_providers
from ..logging import get_logger
from ..config import Config

logger = get_logger(__name__)

# Global config instance for loading API keys
_config = None

def _get_config():
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def _truncate_for_llm(result, max_array_size=20):
    """Truncate large arrays to avoid token overflow."""
    import copy
    import numpy as np
    
    def truncate_array(arr):
        if not isinstance(arr, list) or len(arr) <= max_array_size:
            return arr
        n = len(arr)
        try:
            arr_np = np.array(arr)
            return {
                "total_points": n,
                "first_5": arr[:5],
                "last_5": arr[-5:],
                "min": float(np.min(arr_np)),
                "max": float(np.max(arr_np)),
                "mean": float(np.mean(arr_np)),
            }
        except:
            return {"total_points": n, "first_5": arr[:5], "last_5": arr[-5:]}
    
    def truncate_dict(d):
        if not isinstance(d, dict):
            return d
        result = {}
        for k, v in d.items():
            if isinstance(v, list) and len(v) > max_array_size:
                if v and all(isinstance(x, (int, float)) for x in v[:10]):
                    result[k] = truncate_array(v)
                else:
                    result[k] = v[:max_array_size]
            elif isinstance(v, dict):
                result[k] = truncate_dict(v)
            else:
                result[k] = v
        return result
    
    if result is None:
        return None
    return truncate_dict(copy.deepcopy(result))


@dataclass
class CouncilMember:
    """A member of the LLM council."""
    name: str
    provider_name: str
    provider: Any
    model: str
    emoji: str = "🤖"


@dataclass
class Stage1Response:
    """Stage 1: Individual model response."""
    member_name: str
    provider_name: str
    model: str
    response: str
    anonymous_id: str  # Used for anonymization in Stage 2


@dataclass
class Stage2Ranking:
    """Stage 2: Model rankings of other responses."""
    member_name: str
    provider_name: str
    rankings: List[str]  # Ordered list of anonymous IDs (best to worst)
    reasoning: str


@dataclass
class Stage3Synthesis:
    """Stage 3: Chairman's final synthesis."""
    chairman_name: str
    provider_name: str
    final_response: str
    aggregate_rankings: List[Dict[str, Any]]


@dataclass
class CouncilDeliberation:
    """Complete council deliberation result."""
    stage1_responses: List[Stage1Response]
    stage2_rankings: List[Stage2Ranking]
    stage3_synthesis: Stage3Synthesis
    full_transcript: str
    member_count: int


# Default council configuration
DEFAULT_COUNCIL_CONFIG = {
    "anthropic": {
        "name": "Claude",
        "emoji": "🟣",
        "model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY"
    },
    "gemini": {
        "name": "Gemini",
        "emoji": "🔵",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY"
    },
    "openai": {
        "name": "GPT",
        "emoji": "🟢",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY"
    },
    "deepseek": {
        "name": "DeepSeek",
        "emoji": "🟡",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "qwen": {
        "name": "Qwen",
        "emoji": "🟠",
        "model": "qwen-turbo",
        "env_key": "DASHSCOPE_API_KEY"
    }
}


class MultiLLMCouncil:
    """
    Karpathy-style Multi-LLM Council.

    Uses multiple different LLM providers for deliberation with 3 stages:
    1. Each LLM independently analyzes the question
    2. Each LLM ranks the other models' responses (anonymized)
    3. A chairman synthesizes everything into a final response
    """

    def __init__(
        self,
        providers: Optional[List[str]] = None,
        chairman: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize the Multi-LLM Council.

        Args:
            providers: List of provider names to use (defaults to all available)
            chairman: Provider name for the chairman (defaults to first available)
            progress_callback: Optional callback for progress updates
        """
        self.progress_callback = progress_callback
        self.members: List[CouncilMember] = []
        self.chairman: Optional[CouncilMember] = None

        # Initialize council members from available providers
        self._initialize_members(providers, chairman)

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        logger.info(f"Council progress: {message} ({progress:.0%})")

    def _initialize_members(
        self,
        provider_names: Optional[List[str]] = None,
        chairman_name: Optional[str] = None
    ):
        """Initialize council members from available providers."""
        available = get_available_providers()

        # Use specified providers or all available
        to_use = provider_names or available
        
        # Get config for loading API keys
        config_manager = _get_config()

        for provider_name in to_use:
            if provider_name not in available:
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue

            council_config = DEFAULT_COUNCIL_CONFIG.get(provider_name, {})
            
            # Try to get provider from Config (checks config.yaml and env vars)
            try:
                provider = config_manager.get_provider(provider_name)
                
                if provider:
                    member = CouncilMember(
                        name=council_config.get("name", provider_name.title()),
                        provider_name=provider_name,
                        provider=provider,
                        model=council_config.get("model", "default"),
                        emoji=council_config.get("emoji", "🤖")
                    )
                    self.members.append(member)
                    logger.info(f"Added council member: {member.name} ({provider_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")

        if not self.members:
            raise ValueError("No council members available. Check API keys.")

        # Set chairman (specified or first available)
        if chairman_name:
            for member in self.members:
                if member.provider_name == chairman_name:
                    self.chairman = member
                    break

        if not self.chairman:
            self.chairman = self.members[0]

        logger.info(f"Council initialized with {len(self.members)} members. Chairman: {self.chairman.name}")

    def _generate_anonymous_id(self) -> str:
        """Generate a random anonymous ID for responses."""
        return ''.join(random.choices(string.ascii_uppercase, k=6))

    def _stage1_get_responses(
        self,
        question: str,
        context: Optional[str] = None,
        tool_result: Optional[Dict[str, Any]] = None
    ) -> List[Stage1Response]:
        """
        Stage 1: Get independent responses from each LLM.

        Each model analyzes the question/data independently without seeing
        other models' responses.
        """
        self._report_progress("Stage 1: Gathering independent analyses...", 0.1)

        responses = []

        # Build the prompt
        prompt_parts = []
        if context:
            prompt_parts.append(f"**Context:**\n{context}\n")
        if tool_result:
            # Truncate to avoid token overflow
            truncated = _truncate_for_llm(tool_result)
            prompt_parts.append(f"**Data Analysis Results (truncated):**\n```json\n{json.dumps(truncated, indent=2)}\n```\n")
        prompt_parts.append(f"**Question:** {question}")
        prompt_parts.append("\nProvide a thorough analysis. Be specific with numbers and insights.")

        full_prompt = "\n".join(prompt_parts)

        system_instruction = """You are an expert data analyst on a council of AI experts.
Analyze the provided data and question thoroughly. Provide specific insights,
identify key patterns, and give actionable recommendations.
Be concise but comprehensive. Use numbers and specific observations."""

        # Query each member in parallel
        def get_response(member: CouncilMember) -> Stage1Response:
            try:
                response = member.provider.generate(full_prompt, system_instruction=system_instruction)
                return Stage1Response(
                    member_name=member.name,
                    provider_name=member.provider_name,
                    model=member.model,
                    response=response,
                    anonymous_id=self._generate_anonymous_id()
                )
            except Exception as e:
                logger.error(f"Stage 1 failed for {member.name}: {e}")
                return Stage1Response(
                    member_name=member.name,
                    provider_name=member.provider_name,
                    model=member.model,
                    response=f"[Error: {str(e)}]",
                    anonymous_id=self._generate_anonymous_id()
                )

        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            futures = {executor.submit(get_response, m): m for m in self.members}
            for i, future in enumerate(as_completed(futures)):
                responses.append(future.result())
                progress = 0.1 + (0.25 * (i + 1) / len(self.members))
                self._report_progress(f"Stage 1: Got response {i+1}/{len(self.members)}", progress)

        return responses

    def _stage2_peer_review(
        self,
        question: str,
        stage1_responses: List[Stage1Response]
    ) -> List[Stage2Ranking]:
        """
        Stage 2: Each model ranks the other models' responses.

        Responses are anonymized so models can't play favorites.
        """
        self._report_progress("Stage 2: Peer review and ranking...", 0.4)

        rankings = []

        # Build anonymized responses for ranking
        anonymous_responses = []
        id_to_name = {}  # For debugging
        for resp in stage1_responses:
            anonymous_responses.append({
                "id": resp.anonymous_id,
                "response": resp.response
            })
            id_to_name[resp.anonymous_id] = resp.member_name

        system_instruction = """You are evaluating multiple AI responses to a question.
Rank the responses from BEST to WORST based on:
1. Accuracy of analysis
2. Depth of insights
3. Actionable recommendations
4. Clarity and specificity

Your output must be a JSON object with:
- "ranking": list of response IDs from best to worst
- "reasoning": brief explanation of your ranking"""

        def get_ranking(member: CouncilMember) -> Stage2Ranking:
            # Filter out the member's own response for ranking
            responses_to_rank = [
                r for r in anonymous_responses
                if r["id"] != next(
                    (s.anonymous_id for s in stage1_responses if s.member_name == member.name),
                    None
                )
            ]

            prompt = f"""**Original Question:** {question}

**Responses to Rank:**
"""
            for resp in responses_to_rank:
                prompt += f"\n--- Response [{resp['id']}] ---\n{resp['response']}\n"

            prompt += "\n\nRank these responses from BEST to WORST. Return JSON with 'ranking' (list of IDs) and 'reasoning'."

            try:
                response = member.provider.generate(prompt, system_instruction=system_instruction)

                # Parse JSON from response
                try:
                    # Try to extract JSON from response
                    if "```json" in response:
                        json_str = response.split("```json")[1].split("```")[0]
                    elif "```" in response:
                        json_str = response.split("```")[1].split("```")[0]
                    else:
                        json_str = response

                    parsed = json.loads(json_str.strip())
                    ranking_ids = parsed.get("ranking", [])
                    reasoning = parsed.get("reasoning", "")
                except:
                    # Fallback: just use the response as reasoning
                    ranking_ids = [r["id"] for r in responses_to_rank]
                    reasoning = response

                return Stage2Ranking(
                    member_name=member.name,
                    provider_name=member.provider_name,
                    rankings=ranking_ids,
                    reasoning=reasoning
                )
            except Exception as e:
                logger.error(f"Stage 2 failed for {member.name}: {e}")
                return Stage2Ranking(
                    member_name=member.name,
                    provider_name=member.provider_name,
                    rankings=[],
                    reasoning=f"[Error: {str(e)}]"
                )

        with ThreadPoolExecutor(max_workers=len(self.members)) as executor:
            futures = {executor.submit(get_ranking, m): m for m in self.members}
            for i, future in enumerate(as_completed(futures)):
                rankings.append(future.result())
                progress = 0.4 + (0.25 * (i + 1) / len(self.members))
                self._report_progress(f"Stage 2: Got ranking {i+1}/{len(self.members)}", progress)

        return rankings

    def _compute_aggregate_rankings(
        self,
        stage1_responses: List[Stage1Response],
        stage2_rankings: List[Stage2Ranking]
    ) -> List[Dict[str, Any]]:
        """Compute aggregate rankings using Borda count."""
        scores = {}

        # Initialize scores
        for resp in stage1_responses:
            scores[resp.anonymous_id] = {
                "id": resp.anonymous_id,
                "member": resp.member_name,
                "provider": resp.provider_name,
                "total_score": 0,
                "rankings_received": 0,
                "average_rank": 0
            }

        # Borda count: first place gets N points, second gets N-1, etc.
        for ranking in stage2_rankings:
            n = len(ranking.rankings)
            for i, resp_id in enumerate(ranking.rankings):
                if resp_id in scores:
                    scores[resp_id]["total_score"] += (n - i)
                    scores[resp_id]["rankings_received"] += 1

        # Compute average rank
        for resp_id, data in scores.items():
            if data["rankings_received"] > 0:
                data["average_rank"] = data["total_score"] / data["rankings_received"]

        # Sort by score (highest first)
        sorted_scores = sorted(
            scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )

        return sorted_scores

    def _stage3_synthesis(
        self,
        question: str,
        stage1_responses: List[Stage1Response],
        stage2_rankings: List[Stage2Ranking],
        aggregate_rankings: List[Dict[str, Any]]
    ) -> Stage3Synthesis:
        """
        Stage 3: Chairman synthesizes all responses and rankings.

        The chairman creates a comprehensive final answer incorporating
        the best insights from all models.
        """
        self._report_progress("Stage 3: Chairman synthesizing final response...", 0.7)

        # Build context for chairman
        prompt_parts = [f"**Original Question:** {question}\n"]

        prompt_parts.append("**Council Responses (ranked by peer review):**\n")
        for rank_data in aggregate_rankings:
            resp = next((r for r in stage1_responses if r.anonymous_id == rank_data["id"]), None)
            if resp:
                prompt_parts.append(f"\n### {resp.member_name} (Score: {rank_data['total_score']}):\n{resp.response}\n")

        prompt_parts.append("\n**Peer Review Reasoning:**\n")
        for ranking in stage2_rankings:
            prompt_parts.append(f"- {ranking.member_name}: {ranking.reasoning[:200]}...\n")

        prompt_parts.append("""
**Your Task as Chairman:**
Synthesize all the above responses into a comprehensive final answer that:
1. Incorporates the best insights from the top-ranked responses
2. Resolves any disagreements between models
3. Provides a clear, actionable conclusion
4. Acknowledges areas of uncertainty or disagreement

Be thorough but concise. Structure your response clearly.""")

        system_instruction = f"""You are the Chairman of an AI council, tasked with synthesizing
multiple expert analyses into a final comprehensive response.

The council has {len(self.members)} members. Each has provided their analysis,
and they have ranked each other's responses. Use this information to create
the best possible answer.

Prioritize insights from higher-ranked responses, but include valuable
points from all members. Be balanced and objective."""

        try:
            response = self.chairman.provider.generate(
                "\n".join(prompt_parts),
                system_instruction=system_instruction
            )

            return Stage3Synthesis(
                chairman_name=self.chairman.name,
                provider_name=self.chairman.provider_name,
                final_response=response,
                aggregate_rankings=aggregate_rankings
            )
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            # Fallback to highest-ranked response
            best_response = stage1_responses[0] if stage1_responses else None
            return Stage3Synthesis(
                chairman_name=self.chairman.name,
                provider_name=self.chairman.provider_name,
                final_response=f"[Chairman synthesis failed. Best individual response from {best_response.member_name if best_response else 'unknown'}]:\n\n{best_response.response if best_response else 'No responses available'}",
                aggregate_rankings=aggregate_rankings
            )

    def _build_transcript(
        self,
        question: str,
        stage1_responses: List[Stage1Response],
        stage2_rankings: List[Stage2Ranking],
        stage3_synthesis: Stage3Synthesis
    ) -> str:
        """Build a full transcript of the deliberation."""
        lines = [
            "=" * 60,
            "MULTI-LLM COUNCIL DELIBERATION TRANSCRIPT",
            "=" * 60,
            f"\nQuestion: {question}\n",
            f"Council Members: {', '.join(m.name for m in self.members)}",
            f"Chairman: {self.chairman.name}\n",
            "-" * 40,
            "STAGE 1: INDEPENDENT ANALYSES",
            "-" * 40,
        ]

        for resp in stage1_responses:
            lines.append(f"\n### {resp.member_name} ({resp.provider_name}):")
            lines.append(resp.response)
            lines.append("")

        lines.extend([
            "-" * 40,
            "STAGE 2: PEER REVIEW RANKINGS",
            "-" * 40,
        ])

        for ranking in stage2_rankings:
            lines.append(f"\n{ranking.member_name}'s Rankings: {' > '.join(ranking.rankings)}")
            lines.append(f"Reasoning: {ranking.reasoning}")

        lines.extend([
            "",
            "-" * 40,
            "AGGREGATE RANKINGS (Borda Count)",
            "-" * 40,
        ])

        for i, rank_data in enumerate(stage3_synthesis.aggregate_rankings, 1):
            lines.append(f"{i}. {rank_data['member']} - Score: {rank_data['total_score']}")

        lines.extend([
            "",
            "-" * 40,
            f"STAGE 3: CHAIRMAN'S SYNTHESIS ({stage3_synthesis.chairman_name})",
            "-" * 40,
            "",
            stage3_synthesis.final_response,
            "",
            "=" * 60,
            "END OF TRANSCRIPT",
            "=" * 60,
        ])

        return "\n".join(lines)

    def deliberate(
        self,
        question: str,
        context: Optional[str] = None,
        tool_result: Optional[Dict[str, Any]] = None
    ) -> CouncilDeliberation:
        """
        Run the full 3-stage council deliberation.

        Args:
            question: The question to deliberate on
            context: Optional context (e.g., data description)
            tool_result: Optional tool/skill execution results

        Returns:
            Complete deliberation result with all stages
        """
        logger.info(f"Starting Multi-LLM Council deliberation with {len(self.members)} members")

        # Stage 1: Independent responses
        stage1_responses = self._stage1_get_responses(question, context, tool_result)

        # Stage 2: Peer review rankings
        stage2_rankings = self._stage2_peer_review(question, stage1_responses)

        # Compute aggregate rankings
        aggregate_rankings = self._compute_aggregate_rankings(stage1_responses, stage2_rankings)

        # Stage 3: Chairman synthesis
        stage3_synthesis = self._stage3_synthesis(
            question, stage1_responses, stage2_rankings, aggregate_rankings
        )

        self._report_progress("Council deliberation complete", 1.0)

        # Build full transcript
        transcript = self._build_transcript(
            question, stage1_responses, stage2_rankings, stage3_synthesis
        )

        return CouncilDeliberation(
            stage1_responses=stage1_responses,
            stage2_rankings=stage2_rankings,
            stage3_synthesis=stage3_synthesis,
            full_transcript=transcript,
            member_count=len(self.members)
        )

    def to_dict(self, deliberation: CouncilDeliberation) -> Dict[str, Any]:
        """Convert deliberation result to dictionary for JSON serialization."""
        return {
            "member_count": deliberation.member_count,
            "members": [
                {
                    "name": m.name,
                    "provider": m.provider_name,
                    "model": m.model,
                    "emoji": m.emoji
                }
                for m in self.members
            ],
            "chairman": {
                "name": self.chairman.name,
                "provider": self.chairman.provider_name
            },
            "stage1": [
                {
                    "member": r.member_name,
                    "provider": r.provider_name,
                    "model": r.model,
                    "response": r.response,
                    "anonymous_id": r.anonymous_id
                }
                for r in deliberation.stage1_responses
            ],
            "stage2": [
                {
                    "member": r.member_name,
                    "provider": r.provider_name,
                    "rankings": r.rankings,
                    "reasoning": r.reasoning
                }
                for r in deliberation.stage2_rankings
            ],
            "stage3": {
                "chairman": deliberation.stage3_synthesis.chairman_name,
                "provider": deliberation.stage3_synthesis.provider_name,
                "response": deliberation.stage3_synthesis.final_response,
                "aggregate_rankings": deliberation.stage3_synthesis.aggregate_rankings
            },
            "full_transcript": deliberation.full_transcript
        }
