# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Data Characteristics Analyzer - Analyze time series properties for model selection.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataCharacteristics:
    """Analyzed characteristics of a time series."""
    # Size
    length: int = 0

    # Distribution
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Trend
    has_trend: bool = False
    trend_direction: str = "none"  # "up", "down", "none"
    trend_strength: float = 0.0

    # Seasonality
    has_seasonality: bool = False
    seasonal_period: Optional[int] = None
    seasonal_strength: float = 0.0

    # Stationarity
    is_stationary: bool = True

    # Volatility
    volatility: float = 0.0
    volatility_level: str = "low"  # "low", "medium", "high"

    # Outliers
    outlier_count: int = 0
    outlier_ratio: float = 0.0

    # Frequency
    frequency: Optional[str] = None  # "daily", "weekly", "monthly", etc.

    # Recommendations
    recommended_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "length": self.length,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "has_trend": self.has_trend,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "has_seasonality": self.has_seasonality,
            "seasonal_period": self.seasonal_period,
            "seasonal_strength": self.seasonal_strength,
            "is_stationary": self.is_stationary,
            "volatility": self.volatility,
            "volatility_level": self.volatility_level,
            "outlier_count": self.outlier_count,
            "outlier_ratio": self.outlier_ratio,
            "frequency": self.frequency,
            "recommended_models": self.recommended_models,
        }

    def summary(self) -> str:
        """Get a text summary of characteristics."""
        parts = [
            f"Length: {self.length} points",
            f"Mean: {self.mean:.2f}, Std: {self.std:.2f}",
            f"Trend: {self.trend_direction} (strength: {self.trend_strength:.2f})" if self.has_trend else "No significant trend",
            f"Seasonality: period={self.seasonal_period}" if self.has_seasonality else "No seasonality detected",
            f"Volatility: {self.volatility_level}",
            f"Outliers: {self.outlier_count} ({self.outlier_ratio:.1%})",
        ]
        return "\n".join(parts)


class CharacteristicsAnalyzer:
    """
    Analyze time series data characteristics.

    Uses statistical methods to detect:
    - Trend (linear regression)
    - Seasonality (autocorrelation)
    - Stationarity (variance stability)
    - Outliers (z-score)
    - Volatility (rolling std)
    """

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze(
        self,
        data: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
    ) -> DataCharacteristics:
        """
        Analyze a time series and return its characteristics.

        Args:
            data: DataFrame with time series data
            target_col: Name of the target column
            date_col: Optional name of date column

        Returns:
            DataCharacteristics object
        """
        if target_col not in data.columns:
            logger.error(f"Column '{target_col}' not found")
            return DataCharacteristics()

        values = data[target_col].values

        # Remove NaN values
        values = values[~np.isnan(values)]

        if len(values) == 0:
            logger.warning("No valid values to analyze")
            return DataCharacteristics()

        chars = DataCharacteristics()

        # Basic statistics
        chars.length = len(values)
        chars.mean = float(np.mean(values))
        chars.std = float(np.std(values))
        chars.min_val = float(np.min(values))
        chars.max_val = float(np.max(values))

        # Skewness and Kurtosis
        try:
            from scipy import stats
            chars.skewness = float(stats.skew(values))
            chars.kurtosis = float(stats.kurtosis(values))
        except ImportError:
            chars.skewness = 0.0
            chars.kurtosis = 0.0

        # Trend analysis
        self._analyze_trend(values, chars)

        # Seasonality analysis
        self._analyze_seasonality(values, chars)

        # Stationarity
        self._analyze_stationarity(values, chars)

        # Volatility
        self._analyze_volatility(values, chars)

        # Outliers
        self._analyze_outliers(values, chars)

        # Frequency detection
        if date_col and date_col in data.columns:
            self._analyze_frequency(data, date_col, chars)
        elif hasattr(data.index, 'freq') and data.index.freq:
            chars.frequency = str(data.index.freq)

        # Generate model recommendations
        self._recommend_models(chars)

        logger.info(f"Analyzed {chars.length} points: trend={chars.trend_direction}, seasonality={chars.has_seasonality}")

        return chars

    def _analyze_trend(self, values: np.ndarray, chars: DataCharacteristics) -> None:
        """Analyze trend using linear regression."""
        n = len(values)
        if n < 2:
            return

        x = np.arange(n)

        # Linear regression
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate R-squared as trend strength
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)

        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0

        # Determine if trend is significant
        # Normalize slope by data range
        data_range = np.max(values) - np.min(values)
        if data_range > 0:
            normalized_slope = abs(slope * n) / data_range
        else:
            normalized_slope = 0

        chars.trend_strength = float(r_squared)

        if r_squared > 0.3 and normalized_slope > 0.1:
            chars.has_trend = True
            chars.trend_direction = "up" if slope > 0 else "down"

    def _analyze_seasonality(self, values: np.ndarray, chars: DataCharacteristics) -> None:
        """Analyze seasonality using autocorrelation."""
        n = len(values)
        if n < 14:  # Need at least 2 weeks
            return

        # Check common seasonal periods
        periods_to_check = [7, 12, 24, 30, 52, 365]

        for period in periods_to_check:
            if n < period * 2:
                continue

            # Calculate autocorrelation at this lag
            acf = self._autocorrelation(values, period)

            if acf > 0.3:  # Significant correlation
                chars.has_seasonality = True
                chars.seasonal_period = period
                chars.seasonal_strength = float(acf)
                break

    def _autocorrelation(self, values: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at a specific lag."""
        n = len(values)
        if lag >= n:
            return 0.0

        mean = np.mean(values)
        var = np.var(values)

        if var == 0:
            return 0.0

        x1 = values[:n-lag] - mean
        x2 = values[lag:] - mean

        return float(np.sum(x1 * x2) / ((n - lag) * var))

    def _analyze_stationarity(self, values: np.ndarray, chars: DataCharacteristics) -> None:
        """Check for stationarity using rolling statistics."""
        n = len(values)
        if n < 20:
            return

        window = n // 4

        # Rolling mean
        rolling_mean = pd.Series(values).rolling(window=window).mean().dropna()

        # Rolling std
        rolling_std = pd.Series(values).rolling(window=window).std().dropna()

        # Check if rolling statistics are stable
        mean_var = np.var(rolling_mean)
        std_var = np.var(rolling_std)

        overall_var = np.var(values)

        # If rolling statistics vary significantly, data is non-stationary
        if overall_var > 0:
            chars.is_stationary = (mean_var / overall_var) < 0.1

    def _analyze_volatility(self, values: np.ndarray, chars: DataCharacteristics) -> None:
        """Analyze volatility (standard deviation of returns)."""
        if len(values) < 2:
            return

        # Calculate returns (percent change)
        returns = np.diff(values) / (np.abs(values[:-1]) + 1e-10)

        volatility = float(np.std(returns))
        chars.volatility = volatility

        # Classify volatility level
        if volatility < 0.05:
            chars.volatility_level = "low"
        elif volatility < 0.15:
            chars.volatility_level = "medium"
        else:
            chars.volatility_level = "high"

    def _analyze_outliers(self, values: np.ndarray, chars: DataCharacteristics) -> None:
        """Detect outliers using z-score."""
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            chars.outlier_count = 0
            chars.outlier_ratio = 0.0
            return

        z_scores = np.abs((values - mean) / std)
        outliers = z_scores > 3.0

        chars.outlier_count = int(np.sum(outliers))
        chars.outlier_ratio = chars.outlier_count / len(values)

    def _analyze_frequency(
        self,
        data: pd.DataFrame,
        date_col: str,
        chars: DataCharacteristics
    ) -> None:
        """Detect data frequency from dates."""
        try:
            dates = pd.to_datetime(data[date_col])
            diffs = dates.diff().dropna()

            if len(diffs) == 0:
                return

            median_diff = diffs.median()

            if median_diff <= pd.Timedelta(hours=1):
                chars.frequency = "hourly"
            elif median_diff <= pd.Timedelta(days=1):
                chars.frequency = "daily"
            elif median_diff <= pd.Timedelta(days=7):
                chars.frequency = "weekly"
            elif median_diff <= pd.Timedelta(days=31):
                chars.frequency = "monthly"
            else:
                chars.frequency = "yearly"
        except Exception as e:
            logger.debug(f"Could not determine frequency: {e}")

    def _recommend_models(self, chars: DataCharacteristics) -> None:
        """Generate model recommendations based on characteristics."""
        recommendations = []

        # Based on length
        if chars.length < 100:
            recommendations.extend(["zscore_baseline", "moirai"])
        elif chars.length < 1000:
            recommendations.extend(["moirai", "chronos2", "timesfm"])
        else:
            recommendations.extend(["timesfm", "chronos2", "moirai"])

        # Based on seasonality
        if chars.has_seasonality:
            if "chronos2" not in recommendations:
                recommendations.append("chronos2")

        # Based on volatility
        if chars.volatility_level == "high":
            recommendations.extend(["lag_llama", "tirex", "mad"])

        # Based on trend
        if chars.has_trend and chars.trend_strength > 0.5:
            recommendations.append("timesfm")

        # Deduplicate while preserving order
        seen = set()
        chars.recommended_models = []
        for model in recommendations:
            if model not in seen:
                seen.add(model)
                chars.recommended_models.append(model)
