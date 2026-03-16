# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Multi-Model Ensemble - Run multiple models in parallel and combine results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime

from .selector import ModelSelector, ModelSelection
from .characteristics import CharacteristicsAnalyzer, DataCharacteristics
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Result from a single model."""
    model_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class EnsembleResult:
    """Combined result from multiple models."""
    success: bool
    model_results: Dict[str, ModelResult] = field(default_factory=dict)
    aggregated: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    selection: Optional[ModelSelection] = None
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model_results": {
                name: {
                    "success": r.success,
                    "data": r.data,
                    "error": r.error,
                    "execution_time": r.execution_time,
                }
                for name, r in self.model_results.items()
            },
            "aggregated": self.aggregated,
            "comparison": self.comparison,
            "selection": self.selection.to_dict() if self.selection else None,
            "total_time": self.total_time,
        }


class MultiModelEnsemble:
    """
    Execute multiple models in parallel and aggregate results.

    Supports:
    - Parallel execution for speed
    - Smart model selection
    - Result aggregation (mean, median, voting)
    - Comparison metrics
    """

    def __init__(
        self,
        forecasters: Optional[Dict[str, Any]] = None,
        detectors: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[Any] = None,
        max_workers: int = 4,
    ):
        """
        Initialize the ensemble runner.

        Args:
            forecasters: Available forecaster instances
            detectors: Available detector instances
            llm_provider: Optional LLM for smart selection
            max_workers: Maximum parallel workers
        """
        self.forecasters = forecasters or {}
        self.detectors = detectors or {}
        self.selector = ModelSelector(llm_provider)
        self.analyzer = CharacteristicsAnalyzer()
        self.max_workers = max_workers

    def run_forecasting(
        self,
        data: Any,  # DataFrame
        target_col: str,
        horizon: int = 7,
        models: Optional[List[str]] = None,
        use_smart_selection: bool = True,
    ) -> EnsembleResult:
        """
        Run multiple forecasting models.

        Args:
            data: Time series data
            target_col: Target column name
            horizon: Forecast horizon
            models: Specific models to use (None for smart selection)
            use_smart_selection: Use smart selection if no models specified

        Returns:
            EnsembleResult with all model outputs
        """
        start_time = datetime.now()

        # Analyze characteristics for selection
        characteristics = self.analyzer.analyze(data, target_col)

        # Select models
        selection = None
        if models:
            selected_models = models
        elif use_smart_selection:
            selection = self.selector.select_forecasters(characteristics)
            selected_models = selection.models
        else:
            selected_models = list(self.forecasters.keys())[:4]

        # Filter to available models
        available_models = [m for m in selected_models if m in self.forecasters]

        if not available_models:
            return EnsembleResult(
                success=False,
                selection=selection,
                comparison={"error": "No available forecasting models"},
            )

        # Run models in parallel
        model_results = self._run_parallel(
            available_models,
            self._run_forecaster,
            data=data,
            target_col=target_col,
            horizon=horizon,
        )

        # Aggregate results
        aggregated = self._aggregate_forecasts(model_results)
        comparison = self._compare_forecasts(model_results)

        total_time = (datetime.now() - start_time).total_seconds()

        return EnsembleResult(
            success=len([r for r in model_results.values() if r.success]) > 0,
            model_results=model_results,
            aggregated=aggregated,
            comparison=comparison,
            selection=selection,
            total_time=total_time,
        )

    def run_detection(
        self,
        data: Any,  # DataFrame
        target_col: str,
        threshold: float = 2.0,
        models: Optional[List[str]] = None,
        use_smart_selection: bool = True,
    ) -> EnsembleResult:
        """
        Run multiple anomaly detection models.

        Args:
            data: Time series data
            target_col: Target column name
            threshold: Detection threshold
            models: Specific models to use (None for smart selection)
            use_smart_selection: Use smart selection if no models specified

        Returns:
            EnsembleResult with all model outputs
        """
        start_time = datetime.now()

        # Analyze characteristics
        characteristics = self.analyzer.analyze(data, target_col)

        # Select models
        selection = None
        if models:
            selected_models = models
        elif use_smart_selection:
            selection = self.selector.select_detectors(characteristics)
            selected_models = selection.models
        else:
            selected_models = list(self.detectors.keys())[:3]

        # Filter to available
        available_models = [m for m in selected_models if m in self.detectors]

        if not available_models:
            return EnsembleResult(
                success=False,
                selection=selection,
                comparison={"error": "No available detection models"},
            )

        # Run in parallel
        model_results = self._run_parallel(
            available_models,
            self._run_detector,
            data=data,
            target_col=target_col,
            threshold=threshold,
        )

        # Aggregate results
        aggregated = self._aggregate_detections(model_results)
        comparison = self._compare_detections(model_results)

        total_time = (datetime.now() - start_time).total_seconds()

        return EnsembleResult(
            success=len([r for r in model_results.values() if r.success]) > 0,
            model_results=model_results,
            aggregated=aggregated,
            comparison=comparison,
            selection=selection,
            total_time=total_time,
        )

    def _run_parallel(
        self,
        models: List[str],
        run_func: Callable,
        **kwargs,
    ) -> Dict[str, ModelResult]:
        """Run models in parallel using thread pool."""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_func, model, **kwargs): model
                for model in models
            }

            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"Model {model} failed: {e}")
                    results[model] = ModelResult(
                        model_name=model,
                        success=False,
                        error=str(e),
                    )

        return results

    def _run_forecaster(
        self,
        model_name: str,
        data: Any,
        target_col: str,
        horizon: int,
    ) -> ModelResult:
        """Run a single forecaster."""
        start_time = datetime.now()

        forecaster = self.forecasters.get(model_name)
        if not forecaster:
            return ModelResult(
                model_name=model_name,
                success=False,
                error=f"Forecaster '{model_name}' not available",
            )

        try:
            result = forecaster.forecast(data, target_col=target_col, horizon=horizon)

            if result:
                return ModelResult(
                    model_name=model_name,
                    success=True,
                    data={
                        "predictions": result.predictions.tolist() if hasattr(result.predictions, 'tolist') else result.predictions,
                        "timestamps": [str(t) for t in result.timestamps] if result.timestamps else [],
                        "lower_bound": result.lower_bound.tolist() if hasattr(result.lower_bound, 'tolist') else result.lower_bound,
                        "upper_bound": result.upper_bound.tolist() if hasattr(result.upper_bound, 'tolist') else result.upper_bound,
                    },
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            else:
                return ModelResult(
                    model_name=model_name,
                    success=False,
                    error="No result returned",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        except Exception as e:
            return ModelResult(
                model_name=model_name,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _run_detector(
        self,
        model_name: str,
        data: Any,
        target_col: str,
        threshold: float,
    ) -> ModelResult:
        """Run a single detector."""
        start_time = datetime.now()

        detector = self.detectors.get(model_name)
        if not detector:
            return ModelResult(
                model_name=model_name,
                success=False,
                error=f"Detector '{model_name}' not available",
            )

        try:
            result = detector.detect(data, target_col=target_col, threshold=threshold)

            if result:
                return ModelResult(
                    model_name=model_name,
                    success=True,
                    data={
                        "anomaly_indices": result.anomaly_indices.tolist() if hasattr(result.anomaly_indices, 'tolist') else list(result.anomaly_indices),
                        "anomaly_scores": result.anomaly_scores.tolist() if hasattr(result.anomaly_scores, 'tolist') else list(result.anomaly_scores),
                        "threshold": result.threshold,
                    },
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            else:
                return ModelResult(
                    model_name=model_name,
                    success=False,
                    error="No result returned",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
        except Exception as e:
            return ModelResult(
                model_name=model_name,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    def _aggregate_forecasts(
        self,
        results: Dict[str, ModelResult],
    ) -> Dict[str, Any]:
        """Aggregate forecasts from multiple models."""
        successful = [r for r in results.values() if r.success and "predictions" in r.data]

        if not successful:
            return {}

        predictions = [np.array(r.data["predictions"]) for r in successful]

        # Handle different lengths (use shortest)
        min_len = min(len(p) for p in predictions)
        predictions = [p[:min_len] for p in predictions]
        predictions_array = np.array(predictions)

        return {
            "mean_prediction": np.mean(predictions_array, axis=0).tolist(),
            "median_prediction": np.median(predictions_array, axis=0).tolist(),
            "std_prediction": np.std(predictions_array, axis=0).tolist(),
            "min_prediction": np.min(predictions_array, axis=0).tolist(),
            "max_prediction": np.max(predictions_array, axis=0).tolist(),
            "model_count": len(successful),
        }

    def _aggregate_detections(
        self,
        results: Dict[str, ModelResult],
    ) -> Dict[str, Any]:
        """Aggregate anomaly detections using voting."""
        successful = [r for r in results.values() if r.success and "anomaly_indices" in r.data]

        if not successful:
            return {}

        # Get all detected indices
        all_indices = []
        for r in successful:
            all_indices.extend(r.data["anomaly_indices"])

        if not all_indices:
            return {
                "unanimous_anomalies": [],
                "majority_anomalies": [],
                "any_anomalies": [],
                "model_count": len(successful),
            }

        # Count votes for each index
        from collections import Counter
        votes = Counter(all_indices)

        n_models = len(successful)
        unanimous = [idx for idx, count in votes.items() if count == n_models]
        majority = [idx for idx, count in votes.items() if count > n_models / 2]

        return {
            "unanimous_anomalies": sorted(unanimous),
            "majority_anomalies": sorted(majority),
            "any_anomalies": sorted(set(all_indices)),
            "vote_counts": dict(votes),
            "model_count": n_models,
        }

    def _compare_forecasts(
        self,
        results: Dict[str, ModelResult],
    ) -> Dict[str, Any]:
        """Compare forecast results across models."""
        successful = [r for r in results.values() if r.success and "predictions" in r.data]

        if len(successful) < 2:
            return {"sufficient_models": False}

        predictions = [np.array(r.data["predictions"]) for r in successful]
        min_len = min(len(p) for p in predictions)
        predictions = [p[:min_len] for p in predictions]

        # Calculate agreement (inverse of std)
        std = np.std(predictions, axis=0)
        mean_std = float(np.mean(std))

        # Pairwise correlations
        from itertools import combinations
        correlations = {}
        model_names = [r.model_name for r in successful]

        for (i, name1), (j, name2) in combinations(enumerate(model_names), 2):
            corr = float(np.corrcoef(predictions[i], predictions[j])[0, 1])
            correlations[f"{name1}_vs_{name2}"] = corr

        return {
            "sufficient_models": True,
            "prediction_disagreement": mean_std,
            "correlations": correlations,
            "models_compared": model_names,
        }

    def _compare_detections(
        self,
        results: Dict[str, ModelResult],
    ) -> Dict[str, Any]:
        """Compare detection results across models."""
        successful = [r for r in results.values() if r.success and "anomaly_indices" in r.data]

        if len(successful) < 2:
            return {"sufficient_models": False}

        # Calculate Jaccard similarity between pairs
        from itertools import combinations

        similarities = {}
        model_names = [r.model_name for r in successful]

        for (i, name1), (j, name2) in combinations(enumerate(model_names), 2):
            set1 = set(successful[i].data["anomaly_indices"])
            set2 = set(successful[j].data["anomaly_indices"])

            if set1 or set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
            else:
                jaccard = 1.0  # Both empty = perfect agreement

            similarities[f"{name1}_vs_{name2}"] = jaccard

        return {
            "sufficient_models": True,
            "jaccard_similarities": similarities,
            "models_compared": model_names,
            "mean_agreement": float(np.mean(list(similarities.values()))) if similarities else 0.0,
        }
