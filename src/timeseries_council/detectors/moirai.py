# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Moirai2-based anomaly detector using probabilistic back-prediction.

Uses Moirai2 foundation model to predict each point and flags anomalies
when actual values fall outside the prediction confidence interval.
"""

from typing import Optional, Callable
import pandas as pd
import numpy as np

from .base import BaseDetector
from ..types import DetectionResult, Anomaly, AnomalyType, DetectionMemory
from ..logging import get_logger
from ..utils import get_device

logger = get_logger(__name__)


class MoiraiAnomalyDetector(BaseDetector):
    """
    Anomaly detection using Moirai2 probabilistic back-prediction.
    
    For each point in the series, uses preceding context to predict that point.
    If the actual value falls outside the confidence interval of predictions,
    it is marked as an anomaly. Severity is measured in standard deviations
    beyond the prediction bounds.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = None,
        context_length: int = 64,
        stride: int = 1,
        confidence: float = 95.0,
        num_samples: int = 100
    ):
        """
        Initialize Moirai2 anomaly detector.

        Args:
            model_size: Moirai2 model size ('small', 'base', 'large')
            device: Device for inference ('cpu', 'cuda', or None for auto-detect)
            context_length: Number of preceding points for context
            stride: Predict every N-th point (1=all points, higher=faster)
            confidence: Confidence level in percent (95 → 2.5th-97.5th percentile)
            num_samples: Number of samples for probabilistic prediction
        """
        self.model_size = model_size.lower()
        self.device = get_device(device)
        self.context_length = context_length
        self.stride = max(1, stride)
        self.confidence = confidence
        self.num_samples = num_samples
        
        # Compute percentile bounds from confidence
        # E.g., 95% confidence → lower=2.5, upper=97.5
        self.lower_percentile = (100 - confidence) / 2
        self.upper_percentile = 100 - self.lower_percentile
        
        self._model = None
        self._predictor = None
        
        logger.info(
            f"Initialized MoiraiAnomalyDetector: {model_size} on {self.device}, "
            f"confidence={confidence}% ({self.lower_percentile:.1f}-{self.upper_percentile:.1f} percentile)"
        )

    @property
    def name(self) -> str:
        return f"Moirai2-Anomaly-{self.model_size}"

    @property
    def description(self) -> str:
        return f"Moirai2 back-prediction anomaly detector ({self.confidence}% confidence)"

    def _load_model(self):
        """Lazily load Moirai2 model and create predictor."""
        if self._predictor is not None:
            return True
            
        try:
            import torch
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            
            # Model ID mapping
            MOIRAI_MODELS = {
                "small": "Salesforce/moirai-2.0-R-small",
                "base": "Salesforce/moirai-2.0-R-base",
                "large": "Salesforce/moirai-2.0-R-large",
            }
            
            model_id = MOIRAI_MODELS.get(self.model_size, MOIRAI_MODELS["small"])
            logger.info(f"Loading Moirai2 model: {model_id}")
            
            # Load module from HuggingFace
            module = Moirai2Module.from_pretrained(model_id)
            if module is None:
                raise ValueError("Failed to load Moirai2Module")
            
            # Create forecast wrapper - predict 1 step ahead for back-prediction
            self._model = Moirai2Forecast(
                module=module,
                prediction_length=1,
                context_length=self.context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            
            # Create predictor
            self._predictor = self._model.create_predictor(
                batch_size=1, 
                device=torch.device(self.device)
            )
            
            logger.info("Moirai2 model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Moirai2 model: {e}")
            return False

    def _predict_point(self, context: np.ndarray, timestamp: pd.Timestamp,
                       lower_percentile: float, upper_percentile: float) -> dict:
        """
        Predict a single point given context.
        
        Args:
            context: Preceding values for prediction
            timestamp: Timestamp of the point to predict
            lower_percentile: Lower percentile for confidence interval (e.g., 2.275 for 95.45%)
            upper_percentile: Upper percentile for confidence interval (e.g., 97.725 for 95.45%)
        
        Returns dict with:
            - lower: lower quantile bound
            - upper: upper quantile bound
            - median: median prediction
            - mean: mean prediction
        """
        from gluonts.dataset.common import ListDataset
        
        # Create dataset for this context window
        freq = "D"  # Will be overridden by actual data
        dataset = ListDataset(
            [{"start": timestamp - pd.Timedelta(days=len(context)), "target": context}],
            freq=freq
        )
        
        # Generate predictions
        forecasts = list(self._predictor.predict(dataset))
        if not forecasts:
            return None
            
        forecast = forecasts[0]
        
        # Moirai 2.0 uses QuantileForecast - access quantiles directly
        # Convert percentiles to quantiles (e.g., 2.5% -> 0.025)
        lower_quantile = lower_percentile / 100.0
        upper_quantile = upper_percentile / 100.0
        
        lower_bound = forecast.quantile(lower_quantile)[0]
        upper_bound = forecast.quantile(upper_quantile)[0]
        
        # Estimate std from quantile range
        # For 95% confidence (2.5% to 97.5%), the range is approximately 3.92 standard deviations
        # For other confidence levels, we use the z-score for the upper quantile
        from scipy import stats
        z_score = stats.norm.ppf(upper_quantile)
        estimated_std = (upper_bound - lower_bound) / (2 * z_score) if z_score > 0 else 1.0
        
        return {
            "lower": lower_bound,
            "upper": upper_bound,
            "median": forecast.median[0],
            "mean": forecast.mean[0],
            "std": max(estimated_std, 1e-6)  # Avoid zero std
        }

    def _compute_severity(
        self, 
        actual: float, 
        lower: float, 
        upper: float, 
        pred_std: float
    ) -> tuple:
        """
        Compute severity as standard deviations beyond the bound.
        
        Returns:
            (severity_score, severity_label)
        """
        if pred_std == 0:
            pred_std = 1e-6  # Avoid division by zero
            
        if actual < lower:
            distance = lower - actual
        elif actual > upper:
            distance = actual - upper
        else:
            return (0.0, "normal")
            
        severity_score = distance / pred_std
        
        # Classify severity
        if severity_score >= 4.0:
            severity_label = "extreme"
        elif severity_score >= 2.0:
            severity_label = "severe"
        elif severity_score >= 1.0:
            severity_label = "moderate"
        else:
            severity_label = "mild"
            
        return (severity_score, severity_label)

    def detect(
        self,
        series: pd.Series,
        sensitivity: float = 2.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        memory: Optional[DetectionMemory] = None,
        target_start_idx: Optional[int] = None,
        target_end_idx: Optional[int] = None,
    ) -> DetectionResult:
        """
        Detect anomalies using Moirai back-prediction.
        
        Args:
            series: Time series with DatetimeIndex
            sensitivity: Number of standard deviations for anomaly threshold.
                        Mapped to confidence interval (e.g., 2.0 → 95.45%, 3.0 → 99.73%)
            progress_callback: Optional progress callback
            target_start_idx: If provided, only check for anomalies from this index onward.
                             All preceding data is still used as context for prediction.
                             This enables month/date-range based detection.
            target_end_idx: If provided, only check for anomalies up to this index (exclusive).
                           Combined with target_start_idx, defines the anomaly check range.
            
        Returns:
            DetectionResult with anomalies and metadata
        """
        # Map sensitivity (std devs) to confidence percentage using normal distribution
        # sensitivity=2.0 → 95.45% confidence, sensitivity=3.0 → 99.73% confidence
        from scipy import stats
        confidence = stats.norm.cdf(sensitivity) - stats.norm.cdf(-sensitivity)
        confidence_pct = confidence * 100  # Convert to percentage
        
        # Recalculate percentile bounds based on sensitivity
        lower_percentile = (100 - confidence_pct) / 2
        upper_percentile = 100 - lower_percentile
        
        logger.info(f"Using sensitivity={sensitivity} → confidence={confidence_pct:.2f}% "
                   f"(bounds: {lower_percentile:.2f}%-{upper_percentile:.2f}%)")
        error = self.validate_input(series)
        if error:
            logger.error(f"Validation failed: {error}")
            return DetectionResult(success=False, error=error)
        
        self._report_progress(progress_callback, "Loading Moirai2 model...", 0.05)
        
        # Load model
        if not self._load_model():
            return DetectionResult(
                success=False,
                error="Failed to load Moirai2 model. Ensure uni2ts, gluonts, and torch are installed."
            )
        
        self._report_progress(progress_callback, "Running back-prediction...", 0.1)
        
        try:
            # Ensure DatetimeIndex
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            
            series = series.sort_index()
            values = series.values.astype(np.float32)
            n = len(series)
            
            # Determine the effective start for anomaly checking
            # We need at least context_length points before the check region
            effective_start = target_start_idx if target_start_idx is not None else self.context_length
            effective_end = target_end_idx if target_end_idx is not None else n
            
            # Validate we have enough context
            if effective_start < self.context_length:
                return DetectionResult(
                    success=False,
                    error=f"Need at least {self.context_length} points of context before the target range (got {effective_start})"
                )
            
            # We need at least context_length + 1 points total
            if n <= self.context_length:
                return DetectionResult(
                    success=False,
                    error=f"Need more than {self.context_length} points (got {n})"
                )
            
            anomalies = []
            prediction_details = []
            
            # Iterate over points within the target range
            # Start from max(context_length, effective_start) to ensure enough context
            check_start = max(self.context_length, effective_start)
            points_to_check = range(check_start, effective_end, self.stride)
            total_points = len(list(points_to_check))
            points_to_check = range(check_start, effective_end, self.stride)  # Reset
            
            if total_points == 0:
                return DetectionResult(
                    success=True,
                    anomaly_count=0,
                    anomalies=[],
                    detector_name=self.name,
                    threshold=confidence_pct,
                    metadata={
                        "model_size": self.model_size,
                        "sensitivity": sensitivity,
                        "confidence": confidence_pct,
                        "context_length": self.context_length,
                        "points_checked": 0,
                        "target_range": f"indices {effective_start} to {effective_end}",
                        "message": "No points to check in the target range"
                    }
                )
            
            logger.info(f"Checking {total_points} points in range [{check_start}, {effective_end})")
            
            for i, idx in enumerate(points_to_check):
                # Progress update
                progress = 0.1 + 0.8 * (i / total_points)
                if i % max(1, total_points // 20) == 0:
                    self._report_progress(
                        progress_callback, 
                        f"Checking point {i+1}/{total_points}...", 
                        progress
                    )
                
                # Get context and actual value
                context = values[idx - self.context_length:idx]
                actual = float(values[idx])
                timestamp = series.index[idx]
                
                # Predict this point with dynamic percentile bounds
                pred = self._predict_point(context, timestamp, lower_percentile, upper_percentile)
                if pred is None:
                    continue
                
                # Check if anomaly
                lower, upper = pred["lower"], pred["upper"]
                pred_std = pred["std"]
                
                if actual < lower or actual > upper:
                    severity_score, severity_label = self._compute_severity(
                        actual, lower, upper, pred_std
                    )

                    # Deep memory integration: boost severity when the point
                    # is also far from the baseline distribution.  The final
                    # score is the max of (prediction-based, baseline-based)
                    # so baseline context can only increase severity, never
                    # mask a model-detected anomaly.
                    baseline_note = ""
                    if (
                        memory is not None
                        and memory.baseline_stats.get("mean") is not None
                        and memory.baseline_stats.get("std") is not None
                        and memory.baseline_stats["std"] > 0
                    ):
                        b_mean = memory.baseline_stats["mean"]
                        b_std = memory.baseline_stats["std"]
                        baseline_z = abs(actual - b_mean) / b_std
                        if baseline_z > severity_score:
                            severity_score = baseline_z
                            baseline_note = f" (boosted by baseline z={baseline_z:.1f})"

                    # Determine anomaly type
                    if actual > upper:
                        anomaly_type = AnomalyType.SPIKE
                    else:
                        anomaly_type = AnomalyType.DROP

                    anomalies.append(Anomaly(
                        timestamp=str(timestamp),
                        value=actual,
                        score=severity_score,
                        anomaly_type=anomaly_type,
                        confidence=confidence_pct / 100.0,
                        explanation=f"{severity_label.capitalize()} anomaly: {severity_score:.1f} std devs outside {confidence_pct:.1f}% confidence interval [{lower:.2f}, {upper:.2f}]{baseline_note}"
                    ))
                
                # Store prediction details for metadata
                prediction_details.append({
                    "index": idx,
                    "actual": actual,
                    "predicted_median": pred["median"],
                    "lower": lower,
                    "upper": upper,
                    "is_anomaly": actual < lower or actual > upper
                })
            
            self._report_progress(progress_callback, "Detection complete", 1.0)
            
            # Apply memory context
            anomalies = self._apply_memory(anomalies, memory)

            # Sort anomalies by severity (highest first)
            anomalies.sort(key=lambda a: a.score, reverse=True)

            logger.info(f"Moirai2 detection found {len(anomalies)} anomalies")
            
            return DetectionResult(
                success=True,
                anomaly_count=len(anomalies),
                anomalies=anomalies,
                detector_name=self.name,
                threshold=confidence_pct,
                metadata={
                    "model_size": self.model_size,
                    "sensitivity": sensitivity,
                    "confidence": confidence_pct,
                    "context_length": self.context_length,
                    "stride": self.stride,
                    "points_checked": total_points,
                    "percentile_bounds": [lower_percentile, upper_percentile],
                    "target_range": f"indices {effective_start} to {effective_end}" if target_start_idx is not None else "full series",
                    "baseline_used": (
                        memory is not None
                        and bool(memory.baseline_stats.get("mean") is not None)
                    ),
                    "memory_applied": memory is not None,
                }
            )
            
        except Exception as e:
            logger.error(f"Moirai2 detection failed: {e}")
            import traceback
            return DetectionResult(
                success=False,
                error=str(e),
                metadata={"traceback": traceback.format_exc()}
            )
