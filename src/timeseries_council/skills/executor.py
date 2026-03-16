# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Skill Executor - Execute skills with multi-model support.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import traceback

from .registry import Skill, SkillRegistry, get_registry
from .loader import SkillLoader, load_skills
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SkillResult:
    """Result from skill execution."""
    skill_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    models_used: List[str] = field(default_factory=list)
    thinking: Optional[str] = None  # LLM reasoning about skill selection
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "models_used": self.models_used,
            "thinking": self.thinking,
            "metadata": self.metadata,
        }


@dataclass
class DataContext:
    """Context for skill execution including data and user context."""
    data: Any  # pandas DataFrame or similar
    target_col: str
    date_col: Optional[str] = None
    user_context: Optional[str] = None  # Additional context from user
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        """Check if data is available."""
        return self.data is not None and len(self.data) > 0


class SkillExecutor:
    """
    Execute skills with support for multi-model orchestration.

    Responsibilities:
    - Match queries to skills
    - Execute skills with appropriate models
    - Handle multi-model execution when enabled
    - Provide thinking/reasoning output
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        llm_provider: Optional[Any] = None,
        forecasters: Optional[Dict[str, Any]] = None,
        detectors: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the skill executor.

        Args:
            registry: Skill registry (uses global if not provided)
            llm_provider: LLM provider for reasoning
            forecasters: Available forecaster instances
            detectors: Available detector instances
        """
        self.registry = registry or get_registry()
        self.llm_provider = llm_provider
        self.forecasters = forecasters or {}
        self.detectors = detectors or {}

        # Ensure skills are loaded
        self._ensure_skills_loaded()

        logger.info(f"SkillExecutor initialized with {len(self.registry.list_skills())} skills")

    def _ensure_skills_loaded(self) -> None:
        """Load skills if not already loaded."""
        if not self.registry.list_skills():
            load_skills()
            logger.info("Skills loaded into registry")

    def match_skill(self, query: str) -> Optional[Skill]:
        """
        Find the best matching skill for a query.

        Args:
            query: User's natural language query

        Returns:
            Best matching skill or None
        """
        matches = self.registry.match_query(query)
        if matches:
            logger.debug(f"Query '{query[:50]}...' matched skill: {matches[0].name}")
            return matches[0]
        return None

    def execute(
        self,
        skill_name: str,
        args: Dict[str, Any],
        data_context: Optional[DataContext] = None,
    ) -> SkillResult:
        """
        Execute a skill by name.

        Args:
            skill_name: Name of the skill to execute
            args: Arguments for the skill
            data_context: Data context for execution

        Returns:
            SkillResult with execution results
        """
        start_time = datetime.now()

        skill = self.registry.get(skill_name)
        if not skill:
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill '{skill_name}' not found",
            )

        # Check data requirement
        if skill.requires_data and (not data_context or not data_context.has_data):
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill '{skill_name}' requires data but none provided",
            )

        try:
            # Execute based on skill type
            if skill.multi_model:
                result = self._execute_multi_model(skill, args, data_context)
            else:
                result = self._execute_single(skill, args, data_context)

            result.execution_time = (datetime.now() - start_time).total_seconds()
            return result

        except Exception as e:
            logger.error(f"Error executing skill {skill_name}: {e}")
            logger.debug(traceback.format_exc())
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _execute_single(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: Optional[DataContext],
    ) -> SkillResult:
        """Execute a single-model skill."""
        # Dispatch based on skill name
        if skill.name == "forecasting":
            # Check if multiple models are requested - if so, use multi-model execution
            models = args.get("models") or args.get("forecaster") or args.get("model")
            if isinstance(models, list) and len(models) > 1:
                # Multiple models requested - use multi-model path
                args["models"] = models
                return self._execute_multi_model(skill, args, data_context)
            return self._execute_forecasting(skill, args, data_context)
        elif skill.name == "anomaly_detection":
            return self._execute_detection(skill, args, data_context)
        elif skill.name == "analysis":
            return self._execute_analysis(skill, args, data_context)
        elif skill.name == "simulation":
            return self._execute_simulation(skill, args, data_context)
        elif skill.executor:
            # Use custom executor if defined
            return skill.executor(skill, args, data_context)
        else:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"No executor defined for skill '{skill.name}'",
            )

    def _execute_multi_model(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: Optional[DataContext],
    ) -> SkillResult:
        """
        Execute a skill with multiple models.

        When models aren't specified, uses smart selection.
        """
        models = args.get("models", [])

        # Smart selection if no models specified
        if not models:
            models = self._select_models(skill, data_context)

        # Limit to 3-5 models
        models = models[:5] if len(models) > 5 else models
        if len(models) < 3:
            # Pad with defaults if needed
            models = self._ensure_minimum_models(skill, models, data_context)

        # Execute with each model
        results = {}
        errors = []

        for model_name in models:
            try:
                if skill.name == "forecasting":
                    result = self._run_forecaster(model_name, args, data_context)
                elif skill.name == "anomaly_detection":
                    result = self._run_detector(model_name, args, data_context)
                else:
                    continue

                if result:
                    # Wrap in format expected by frontend: {success: True, data: {...}}
                    results[model_name] = {
                        "success": True,
                        "data": result
                    }
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                logger.warning(f"Model {model_name} failed: {e}")

        if not results:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"All models failed: {'; '.join(errors)}",
                models_used=models,
            )

        return SkillResult(
            skill_name=skill.name,
            success=True,
            data={
                "model_results": results,
                "comparison": self._compare_results(results),
            },
            models_used=list(results.keys()),
            metadata={"errors": errors} if errors else {},
        )

    def _select_models(
        self,
        skill: Skill,
        data_context: Optional[DataContext],
    ) -> List[str]:
        """
        Smart model selection based on data characteristics.

        Uses LLM if available, otherwise falls back to heuristics.
        """
        if skill.name == "forecasting":
            # Default forecasting models
            available = list(self.forecasters.keys()) if self.forecasters else [
                "moirai", "chronos", "timesfm", "zscore_baseline"
            ]
            # Prioritize based on data characteristics
            if data_context and data_context.has_data:
                data_len = len(data_context.data)
                if data_len < 100:
                    # Short series - use simpler models
                    return ["zscore_baseline", "moirai", "chronos"][:3]
            return available[:4]

        elif skill.name == "anomaly_detection":
            # Default detection models
            available = list(self.detectors.keys()) if self.detectors else [
                "zscore", "mad", "isolation_forest", "lof"
            ]
            return available[:4]

        return []

    def _ensure_minimum_models(
        self,
        skill: Skill,
        models: List[str],
        data_context: Optional[DataContext],
    ) -> List[str]:
        """Ensure at least 3 models are selected."""
        if len(models) >= 3:
            return models

        defaults = self._select_models(skill, data_context)
        for default in defaults:
            if default not in models:
                models.append(default)
            if len(models) >= 3:
                break
        return models

    def _run_forecaster(
        self,
        model_name: str,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> Optional[Dict[str, Any]]:
        """Run a forecaster model."""
        forecaster = self.forecasters.get(model_name)
        if not forecaster:
            # Try to create it
            try:
                from ..forecasters import create_forecaster
                forecaster = create_forecaster(model_name)
                if forecaster:
                    self.forecasters[model_name] = forecaster
            except ImportError:
                pass

        if not forecaster:
            logger.warning(f"Forecaster {model_name} not available")
            return None

        horizon = args.get("horizon", 7)
        result = forecaster.forecast(
            data_context.data,
            target_col=data_context.target_col,
            horizon=horizon,
        )

        if result:
            return {
                "predictions": result.predictions.tolist() if hasattr(result.predictions, 'tolist') else result.predictions,
                "timestamps": [str(t) for t in result.timestamps] if result.timestamps else [],
                "lower_bound": result.lower_bound.tolist() if hasattr(result.lower_bound, 'tolist') else result.lower_bound,
                "upper_bound": result.upper_bound.tolist() if hasattr(result.upper_bound, 'tolist') else result.upper_bound,
                "model_name": result.model_name,
            }
        return None

    def _run_detector(
        self,
        model_name: str,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> Optional[Dict[str, Any]]:
        """Run an anomaly detector."""
        detector = self.detectors.get(model_name)
        if not detector:
            # Try to create it
            try:
                from ..detectors import create_detector
                detector = create_detector(model_name)
                if detector:
                    self.detectors[model_name] = detector
            except ImportError:
                pass

        if not detector:
            logger.warning(f"Detector {model_name} not available")
            return None

        threshold = args.get("threshold", 2.0)
        result = detector.detect(
            data_context.data,
            target_col=data_context.target_col,
            threshold=threshold,
        )

        if result:
            return {
                "anomaly_indices": result.anomaly_indices.tolist() if hasattr(result.anomaly_indices, 'tolist') else result.anomaly_indices,
                "anomaly_scores": result.anomaly_scores.tolist() if hasattr(result.anomaly_scores, 'tolist') else result.anomaly_scores,
                "threshold": result.threshold,
                "detector_name": result.detector_name,
            }
        return None

    def _compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from multiple models."""
        comparison = {
            "model_count": len(results),
            "models": list(results.keys()),
        }

        # Extract data from wrapped format {success: True, data: {...}}
        def get_data(r):
            if isinstance(r, dict) and "data" in r:
                return r["data"]
            return r

        # For forecasting, compare predictions
        result_data = [get_data(r) for r in results.values()]
        if all("predictions" in d for d in result_data):
            import numpy as np
            predictions = [np.array(d["predictions"]) for d in result_data]
            if predictions:
                comparison["mean_prediction"] = np.mean(predictions, axis=0).tolist()
                comparison["std_prediction"] = np.std(predictions, axis=0).tolist()
                comparison["prediction_agreement"] = float(np.mean(np.std(predictions, axis=0)))

        # For detection, compare anomaly indices
        if all("anomaly_indices" in d for d in result_data):
            all_indices = [set(d["anomaly_indices"]) for d in result_data]
            if all_indices:
                common = set.intersection(*all_indices) if all_indices else set()
                union = set.union(*all_indices) if all_indices else set()
                comparison["common_anomalies"] = list(common)
                comparison["all_anomalies"] = list(union)
                comparison["agreement_ratio"] = len(common) / len(union) if union else 1.0

        return comparison

    def _execute_forecasting(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> SkillResult:
        """Execute forecasting skill with a single model."""
        # Support both 'model' and 'forecaster' parameter names
        model_name = args.get("model") or args.get("forecaster") or "zscore_baseline"
        
        # If it's a list with one item, extract it
        if isinstance(model_name, list):
            if len(model_name) == 1:
                model_name = model_name[0]
            else:
                # Multiple models - redirect to multi-model execution
                args["models"] = model_name
                return self._execute_multi_model(skill, args, data_context)
        
        result = self._run_forecaster(model_name, args, data_context)

        if result:
            return SkillResult(
                skill_name=skill.name,
                success=True,
                data=result,
                models_used=[model_name],
            )
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=f"Forecaster {model_name} failed or not available",
        )

    def _execute_detection(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> SkillResult:
        """Execute anomaly detection skill with a single model."""
        model_name = args.get("model", "zscore")
        result = self._run_detector(model_name, args, data_context)

        if result:
            return SkillResult(
                skill_name=skill.name,
                success=True,
                data=result,
                models_used=[model_name],
            )
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=f"Detector {model_name} failed or not available",
        )

    def _execute_analysis(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> SkillResult:
        """Execute analysis skill."""
        import numpy as np
        import pandas as pd

        df = data_context.data
        target = data_context.target_col

        if target not in df.columns:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"Target column '{target}' not found",
            )

        values = df[target].values

        analysis = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

        # Trend analysis
        if len(values) > 1:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            analysis["trend_slope"] = float(slope)
            analysis["trend_direction"] = "increasing" if slope > 0 else "decreasing"

        return SkillResult(
            skill_name=skill.name,
            success=True,
            data=analysis,
        )

    def _execute_simulation(
        self,
        skill: Skill,
        args: Dict[str, Any],
        data_context: DataContext,
    ) -> SkillResult:
        """Execute simulation skill using what_if_simulation."""
        from ..tools.simulation import what_if_simulation

        # Get the csv path from metadata (set by orchestrator)
        csv_path = data_context.metadata.get("csv_path")
        target = data_context.target_col

        if not csv_path:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error="CSV path not available in data context",
            )

        # Extract parameters
        scale_factor = args.get("scale_factor", 1.2)
        apply_to_last = args.get("apply_to_last")
        horizon = args.get("horizon", 14)

        # Call the actual what_if_simulation tool
        result = what_if_simulation(
            csv_path=csv_path,
            target_col=target,
            scale_factor=scale_factor,
            horizon=horizon,
            apply_to_last=apply_to_last
        )

        if result.get("success"):
            return SkillResult(
                skill_name=skill.name,
                success=True,
                data=result,
            )
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=result.get("error", "Simulation failed"),
        )

    def execute_from_query(
        self,
        query: str,
        data_context: Optional[DataContext] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """
        Execute a skill based on natural language query.

        Args:
            query: User's query
            data_context: Data context
            args: Optional override arguments

        Returns:
            SkillResult from matched skill
        """
        skill = self.match_skill(query)

        if not skill:
            return SkillResult(
                skill_name="unknown",
                success=False,
                error="No matching skill found for query",
                thinking=f"Could not match query '{query[:100]}' to any skill",
            )

        # Build args from query context
        execution_args = args or {}

        # Add user context if provided
        if data_context and data_context.user_context:
            execution_args["user_context"] = data_context.user_context

        result = self.execute(skill.name, execution_args, data_context)
        result.thinking = f"Matched query to skill '{skill.name}' with confidence"

        return result
