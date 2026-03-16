# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Advanced 3-stage LLM Council orchestration.
Adapted from Karpathy's llm-council: https://github.com/karpathy/llm-council

Stage 1: Collect individual responses from all council models
Stage 2: Each model ranks the anonymized responses of others
Stage 3: Chairman model synthesizes final response
"""

import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from ..providers.base import BaseLLMProvider
from ..logging import get_logger
from ..types import ProgressStage

logger = get_logger(__name__)


@dataclass
class CouncilConfig:
    """Configuration for the advanced council."""
    council_models: Dict[str, BaseLLMProvider] = field(default_factory=dict)
    chairman_model: str = None
    chairman_provider: BaseLLMProvider = None


class AdvancedCouncil:
    """
    Advanced LLM Council with 3-stage deliberation process.

    Stage 1: First Opinions - All models respond to the query
    Stage 2: Peer Review - Models rank each other's responses
    Stage 3: Synthesis - Chairman produces final answer
    """

    def __init__(
        self,
        council_providers: Dict[str, BaseLLMProvider],
        chairman_name: str = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize the advanced council.

        Args:
            council_providers: Dict mapping model names to provider instances
            chairman_name: Name of the chairman model (defaults to first model)
            progress_callback: Optional callback for progress updates
        """
        self.council_providers = council_providers
        self.model_names = list(council_providers.keys())
        self.progress_callback = progress_callback

        # Set chairman (defaults to first model)
        self.chairman_name = chairman_name or self.model_names[0]
        if self.chairman_name not in council_providers:
            self.chairman_name = self.model_names[0]

        self.chairman_provider = council_providers[self.chairman_name]

        logger.info(f"Initialized AdvancedCouncil with {len(council_providers)} models")
        logger.info(f"Chairman: {self.chairman_name}")

    def _report_progress(self, stage: ProgressStage, message: str, progress: float):
        """Report progress through callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, message, min(1.0, max(0.0, progress)))
        logger.debug(f"Progress [{stage.value}]: {progress:.0%} - {message}")

    def run_sync(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Run the full 3-stage council process synchronously.

        Args:
            user_query: The user's question
            context: Optional context about the data

        Returns:
            Dict with stage1, stage2, stage3 results and metadata
        """
        logger.info("Running council deliberation synchronously")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.run_async(user_query, context)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.run_async(user_query, context))
        except RuntimeError:
            return asyncio.run(self.run_async(user_query, context))

    async def run_async(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Run the full 3-stage council process asynchronously.

        Args:
            user_query: The user's question
            context: Optional context about the data

        Returns:
            Dict with stage1, stage2, stage3 results and metadata
        """
        full_query = f"{context}\n\n{user_query}" if context else user_query

        # Stage 1: Collect individual responses
        self._report_progress(ProgressStage.COUNCIL_STAGE_1, "Collecting first opinions...", 0.1)
        logger.info("Council Stage 1: Collecting first opinions")
        stage1_results = await self._stage1_collect(full_query)

        if not stage1_results:
            logger.error("All models failed in Stage 1")
            return {
                "stage1": [],
                "stage2": [],
                "stage3": {"model": "error", "response": "All models failed to respond."},
                "metadata": {}
            }

        # Stage 2: Collect rankings
        self._report_progress(ProgressStage.COUNCIL_STAGE_2, "Peer review and ranking...", 0.4)
        logger.info("Council Stage 2: Peer review and ranking")
        stage2_results, label_to_model = await self._stage2_rank(user_query, stage1_results)

        # Calculate aggregate rankings
        aggregate_rankings = self._calculate_aggregate_rankings(stage2_results, label_to_model)

        # Stage 3: Synthesize final answer
        self._report_progress(ProgressStage.COUNCIL_STAGE_3, "Chairman synthesizing...", 0.7)
        logger.info("Council Stage 3: Chairman synthesizing final answer")
        stage3_result = await self._stage3_synthesize(
            user_query, stage1_results, stage2_results, aggregate_rankings
        )

        self._report_progress(ProgressStage.COMPLETE, "Council deliberation complete", 1.0)

        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": stage3_result,
            "metadata": {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings,
                "chairman": self.chairman_name
            }
        }

    async def _stage1_collect(self, query: str) -> List[Dict[str, Any]]:
        """Stage 1: Collect individual responses from all council models."""
        tasks = []
        for name, provider in self.council_providers.items():
            tasks.append(self._query_model(name, provider, query))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for name, response in zip(self.model_names, responses):
            if isinstance(response, Exception):
                logger.warning(f"Model {name} failed: {response}")
                continue
            if response:
                results.append({
                    "model": name,
                    "response": response,
                    "provider": self.council_providers[name].provider_name
                })
                logger.debug(f"Model {name} responded successfully")

        logger.info(f"Stage 1 complete: {len(results)}/{len(self.model_names)} models responded")
        return results

    async def _stage2_rank(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Stage 2: Each model ranks the anonymized responses."""
        # Create anonymized labels
        labels = [chr(65 + i) for i in range(len(stage1_results))]
        label_to_model = {
            f"Response {label}": result['model']
            for label, result in zip(labels, stage1_results)
        }

        # Build ranking prompt
        responses_text = "\n\n".join([
            f"Response {label}:\n{result['response']}"
            for label, result in zip(labels, stage1_results)
        ])

        ranking_prompt = f"""You are evaluating different AI responses to the following question:

Question: {user_query}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. Evaluate each response individually - what does it do well? What does it miss?
2. Then provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as:
- Start with "FINAL RANKING:" (all caps, with colon)
- List responses from best to worst as numbered list
- Format: "1. Response A" (number, period, space, label)

Example format:
Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth...

FINAL RANKING:
1. Response A
2. Response B
3. Response C

Now provide your evaluation and ranking:"""

        # Get rankings from all models
        tasks = []
        for name, provider in self.council_providers.items():
            tasks.append(self._query_model(name, provider, ranking_prompt))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for name, response in zip(self.model_names, responses):
            if isinstance(response, Exception) or not response:
                logger.warning(f"Model {name} failed to rank")
                continue

            parsed = self._parse_ranking(response)
            results.append({
                "model": name,
                "ranking": response,
                "parsed_ranking": parsed,
                "provider": self.council_providers[name].provider_name
            })
            logger.debug(f"Model {name} ranked successfully")

        logger.info(f"Stage 2 complete: {len(results)} models ranked")
        return results, label_to_model

    async def _stage3_synthesize(
        self,
        user_query: str,
        stage1_results: List[Dict[str, Any]],
        stage2_results: List[Dict[str, Any]],
        aggregate_rankings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stage 3: Chairman synthesizes the final response."""
        # Build context for chairman
        stage1_text = "\n\n".join([
            f"**{result['model']}** ({result.get('provider', 'unknown')}):\n{result['response']}"
            for result in stage1_results
        ])

        rankings_text = "\n".join([
            f"{i+1}. {r['model']} (avg rank: {r['average_rank']:.2f})"
            for i, r in enumerate(aggregate_rankings)
        ])

        chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question and ranked each other's work.

**Original Question:** {user_query}

**STAGE 1 - Individual Responses:**
{stage1_text}

**AGGREGATE PEER RANKINGS (best to worst):**
{rankings_text}

**Your Task as Chairman:**
Synthesize all responses and rankings into a single, comprehensive, accurate answer. Consider:
- The quality and insights from each response
- The peer rankings and what they reveal
- Areas of agreement and disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

        response = await self._query_model(
            self.chairman_name,
            self.chairman_provider,
            chairman_prompt
        )

        if not response:
            logger.warning("Chairman failed, using fallback")
            if aggregate_rankings:
                top_model = aggregate_rankings[0]['model']
                for r in stage1_results:
                    if r['model'] == top_model:
                        return {
                            "model": top_model,
                            "response": f"[Chairman failed, using top-ranked response]\n\n{r['response']}",
                            "is_fallback": True
                        }
            return {
                "model": "error",
                "response": "Unable to generate final synthesis.",
                "is_fallback": True
            }

        logger.info("Stage 3 complete: Chairman synthesized response")
        return {
            "model": self.chairman_name,
            "response": response,
            "is_fallback": False
        }

    async def _query_model(
        self,
        name: str,
        provider: BaseLLMProvider,
        prompt: str
    ) -> Optional[str]:
        """Query a single model asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: provider.generate(prompt, temperature=0.3)
            )
            return response
        except Exception as e:
            logger.error(f"Error querying {name}: {e}")
            return None

    def _parse_ranking(self, text: str) -> List[str]:
        """Parse the FINAL RANKING section from model's response."""
        if "FINAL RANKING:" in text:
            parts = text.split("FINAL RANKING:")
            if len(parts) >= 2:
                ranking_section = parts[1]
                matches = re.findall(r'\d+\.\s*(Response [A-Z])', ranking_section)
                if matches:
                    return matches

        return re.findall(r'Response [A-Z]', text)

    def _calculate_aggregate_rankings(
        self,
        stage2_results: List[Dict[str, Any]],
        label_to_model: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Calculate aggregate rankings across all models."""
        from collections import defaultdict

        model_positions = defaultdict(list)

        for ranking in stage2_results:
            parsed = ranking.get('parsed_ranking', [])
            for position, label in enumerate(parsed, start=1):
                if label in label_to_model:
                    model_name = label_to_model[label]
                    model_positions[model_name].append(position)

        aggregate = []
        for model, positions in model_positions.items():
            if positions:
                avg_rank = sum(positions) / len(positions)
                # Score: lower rank is better, convert to higher score
                score = len(self.model_names) + 1 - avg_rank
                aggregate.append({
                    "model": model,
                    "average_rank": round(avg_rank, 2),
                    "score": round(score, 2),
                    "rankings_count": len(positions)
                })

        aggregate.sort(key=lambda x: x['average_rank'])
        return aggregate


def run_full_council(
    providers: Dict[str, BaseLLMProvider],
    query: str,
    context: str = "",
    chairman_name: str = None
) -> Dict[str, Any]:
    """
    Run the complete 3-stage council process.

    Args:
        providers: Dict mapping model names to provider instances
        query: The user's question
        context: Optional context about the data
        chairman_name: Name of chairman model (defaults to first)

    Returns:
        Dict with stage1, stage2, stage3 results and metadata
    """
    council = AdvancedCouncil(providers, chairman_name)
    return council.run_sync(query, context)
