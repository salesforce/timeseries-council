# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Time Series Council Orchestrator
================================
Main orchestration logic: LLM → Skill/Tool → Execution → LLM → Response
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from .providers.base import BaseLLMProvider
from .tools import TOOLS
from .skills import (
    SkillExecutor, SkillResult, DataContext,
    get_registry, load_skills, DynamicSkillGenerator
)
from .logging import get_logger
from .exceptions import ToolError, OrchestratorError
from .types import ProgressStage

logger = get_logger(__name__)


# System prompt with comprehensive few-shot examples for tool calling
SYSTEM_PROMPT = """You are a Time Series Analysis Assistant powered by advanced ML models.

You help users analyze time series data by calling specialized tools. Based on the user's question, decide which tool to call.

## Multi-Model Approach

This system uses an ensemble of 3-5 models for robust analysis:
- **Forecasting**: Combines multiple forecasters (Moirai, Chronos, TimesFM, TiRex, baseline) for ensemble predictions
- **Anomaly Detection**: Runs multiple detectors (Z-Score, MAD, Isolation Forest, LOF, Moirai) with consensus voting

When summarizing results, ALWAYS explain:
1. Which models were used and why they were selected
2. How the models complement each other
3. The confidence level based on model agreement

## Available Tools

1. **run_forecast** - Predict future values using AI models
   - Use when: user asks about predictions, forecasts, future values, what will happen, next week/month
   - Args:
     - csv_path, target_col (auto-injected)
     - horizon: number of steps to predict (e.g., 720 for 1 day at 2-min freq)
     - forecaster: "multi" (default), or specific model ("moirai", "chronos", "timesfm", "tirex", "lag-llama"), or ARRAY ["moirai", "chronos"]
     - model_size: "small" (default, fast), "base" (balanced), "large" (accurate but slow)
     - context_length: historical points to use (default 168, increase for more context)
   - User can say: "forecast next week with large model", "predict 500 steps using chronos base"

2. **describe_series** - Get statistics and trend analysis
   - Use when: user asks about trends, statistics, summary, what happened, describe, overview
   - Args:
     - csv_path, target_col (auto-injected)
     - window: rolling window size for trend analysis (default 7)
   - User can say: "describe with 30-day window", "give me stats with window of 100"

3. **detect_anomalies** - Find unusual spikes or drops using ML detectors OR custom business rules
   - Use when: user asks about anomalies, outliers, unusual values, spikes, drops, strange
   - **IMPORTANT conversational flow**: If a user asks to "find anomalies" WITHOUT specifying a method, you MUST ask them: "Would you like me to use standard ML models (like Isolation Forest, Moirai, etc.), or do you want to define your own custom business rules? For example, you could say: *Find anomalies where values spike above 2000 and shake within a 200 amplitude band for 5 points.*"
   - Args:
     - csv_path, target_col (auto-injected)
     - detector: "multi" (default, runs 3-5 ML detectors), "rule" (for custom business logic), or specific ML models: "zscore", "mad", "isolation-forest", "lof", "windstats", "moirai", "ecod", "copod", "hbos", "knn", "ocsvm", "loda"
     - custom_prompt: the custom rule logic in natural language (REQUIRED if detector is "rule")
     - sensitivity: threshold (default 2.0, higher = fewer anomalies, lower = more sensitive)
     - confidence: for moirai detector, confidence level in % (e.g., 99 for 99% confidence interval, default 95)
     - target_month: filter to specific month (e.g., "september", "sep", or 9)
     - target_year: specific year if multiple years have the target month
     - start_date: start date for date range (e.g., "15th Aug", "Aug 15", "2024-08-15")
     - end_date: end date for date range (e.g., "3rd Sept", "Sept 3", "2024-09-03")
     - contamination: expected % of outliers (0.01-0.5, for isolation-forest/lof, e.g., 0.05 = 5%)
     - n_estimators: number of trees for isolation forest (default 100, more = more accurate)
   - **IMPORTANT for Moirai**: When using Moirai with a date range/month, all preceding data is used as context for prediction, and anomalies are checked only within the target range
   - User can say: "find anomalies using moirai for september", "detect anomalies with moirai at 99% confidence", "find anomalies using rule: values above 2000"
   - **DATE RANGE**: User can say: "detect anomalies from 15th aug to 3rd sept" → use start_date and end_date


4. **decompose_series** - Decompose into trend, seasonal, and residual
   - Use when: user asks about seasonality, seasonal patterns, decomposition, weekly/monthly patterns
   - Args:
     - csv_path, target_col (auto-injected)
     - period: seasonality period (default 7 for weekly; 24 for hourly data with daily pattern; 720 for 2-min data with daily pattern)
     - model: "additive" (default) or "multiplicative"
   - User can say: "decompose with period 720", "show weekly decomposition with multiplicative model"

5. **compare_series** - Compare multiple columns, correlation analysis
   - Use when: user asks about comparing columns, correlations, relationships between metrics
   - Args:
     - csv_path (auto-injected)
     - columns: list of column names (optional, defaults to all numeric)
   - User can say: "compare column_a and column_b"

6. **what_if_simulation** - Scenario analysis, simulate changes
   - Use when: user asks "what if", scenario planning, impact of changes, simulation, increase/decrease
   - Args:
     - csv_path, target_col (auto-injected)
     - scale_factor: how much to scale values (e.g., 1.2 = 20% increase, 0.8 = 20% decrease)
     - apply_to_last: IMPORTANT - if user specifies a range like "last 30 points" or "last month", set this to the number of points (e.g., 30)
     - horizon: steps to simulate (default 14)
   - User can say: "what if values increase by 50%", "increase the last 30 points by 10%", "simulate 30% drop for last month"

7. **compare_periods** - Compare statistics across time periods (months or quarters)
   - Use when: user asks to COMPARE months/periods/quarters, asks which month is highest/lowest, asks about differences between periods
   - Triggers: "compare March and April", "which month had highest", "difference between Jan and Feb", "compare periods", "compare q1 and q3"
   - Args:
     - csv_path, target_col (auto-injected)
     - periods: list of periods to compare. For quarters, pass as nested lists of months:
       - Single months: ["March", "April"]
       - Quarters: [["January", "February", "March"], ["July", "August", "September"]] for Q1 vs Q3
     - period1: first period (alternative to periods list)
     - period2: second period (alternative to periods list)
   - **Important**: For quarter comparisons like "compare q1 and q3", expand to nested month lists
   - Returns: mean, median, min, max, trend, change% for each period + comparison insights
   - User can say: "compare march and april", "which month had highest average", "compare q1 and q3"

8. **backtest_forecast** - Test forecasts on historical data with custom windows, compare predictions vs actuals
   - Use when: user wants to validate forecasts, test accuracy, compare with actual values, use specific data ranges for prediction
   - Triggers: "use first X to predict", "predict last N and compare", "backtest", "validate forecast", "test on historical", "forecast [month]", "predict [month] to [month]", "forecast jan 4 to jan 11"
   - Args:
     - csv_path, target_col (auto-injected)
     - context_end: where context window ends. Can be: integer (e.g., 5), "first 5", "first 5 months", "first half", "80%"
     - horizon: steps to predict after context (default: predict all remaining points)
     - target_month: target month to forecast. Can be: integer (1-12) or string ("September", "Sep", "March 2024")
     - target_year: specific year if multiple years have the target month
     - start_month: START of month range (for ranges like "Sept to Dec"). Integer (1-12) or string ("September", "Sep")
     - end_month: END of month range (for ranges like "Sept to Dec"). Integer (1-12) or string ("December", "Dec")
     - start_date: START date for SPECIFIC date ranges (for ranges like "Jan 4 to Jan 11"). String ("Jan 4", "2026-01-04")
     - end_date: END date for SPECIFIC date ranges (for ranges like "Jan 4 to Jan 11"). String ("Jan 11", "2026-01-11")
     - compare_actual: whether to compare with actual values and show metrics (default True)
   - Returns: predictions, actuals (if available), and accuracy metrics (MAE, MAPE, SMAPE, RMSE)
   - If target_month exists in multiple years, returns clarification options - ask user which year
   - User can say: "forecast September", "predict March using moirai", "backtest October 2024"
   - **MONTH RANGE**: User can say: "predict sept to dec", "forecast September to December" → use start_month and end_month
   - **SPECIFIC DATE RANGE**: User can say: "predict jan 4 to jan 11", "forecast 2026-01-04 to 2026-01-11" → use start_date and end_date

## Month-Based Filtering (applies to multiple tools)

When user mentions a specific month for analysis:
- Extract target_month: "september" → target_month=9 or target_month="september"
- Extract target_year if specified: "september 2024" → target_month=9, target_year=2024
- For backtest_forecast: use all data before target month as context, predict the month
- For detect_anomalies: filter to only that month's data (use target_month arg)
- For what_if_simulation: apply simulation only to that month (use target_month arg)

## Date Range Parsing (IMPORTANT for backtest_forecast)

When user mentions a RANGE of months like "september to december" or "sept to dec":
- Use backtest_forecast with start_month and end_month (NOT target_month)
- Extract start_month: "sept to dec" → start_month="september"
- Extract end_month: "sept to dec" → end_month="december"
- Example: "predict sept to dec using moirai" → {"tool": "backtest_forecast", "args": {"start_month": "september", "end_month": "december", "forecaster": "moirai"}}

When user mentions SPECIFIC DATES like "jan 4 to jan 11" or "january 4 to january 11":
- Use backtest_forecast with start_date and end_date (NOT start_month/end_month or target_month)
- Extract start_date: "jan 4 to jan 11" → start_date="jan 4"
- Extract end_date: "jan 4 to jan 11" → end_date="jan 11"
- Example: "predict jan 4 to jan 11 using moirai" → {"tool": "backtest_forecast", "args": {"start_date": "jan 4", "end_date": "jan 11", "forecaster": "moirai"}}
- Do NOT use run_forecast for date range requests - use backtest_forecast with start_date/end_date

## Response Format

When you need to call a tool, respond with ONLY a JSON block:
```json
{"tool": "tool_name", "args": {"arg1": "value1", ...}}
```

**IMPORTANT**: Extract any user-specified parameters from their message:
- If user says "contamination 0.1" → args: {"contamination": 0.1}
- If user says "large model" → args: {"model_size": "large"}
- If user says "sensitivity 1.5" → args: {"sensitivity": 1.5}
- If user says "period 720" → args: {"period": 720}
- If user says "last 30 points" or "last month" (which is typically ~30 days) → args: {"apply_to_last": 30}
- If user says "increase by 10%" → args: {"scale_factor": 1.1}
- If user says "forecast September" → args: {"target_month": "september"} (for backtest_forecast)
- If user says "anomalies in March 2024" → args: {"target_month": 3, "target_year": 2024}
- **MONTH RANGE**: If user says "predict sept to dec" → args: {"start_month": "september", "end_month": "december"} (use backtest_forecast)
- **SPECIFIC DATE RANGE**: If user says "predict jan 4 to jan 11" → args: {"start_date": "jan 4", "end_date": "jan 11"} (use backtest_forecast, NOT run_forecast)
- **ANOMALY DATE RANGE**: If user says "detect anomalies from 15th aug to 3rd sept" → args: {"start_date": "15th aug", "end_date": "3rd sept"} (for detect_anomalies)
- **ANOMALY WITH MOIRAI**: If user says "find anomalies using moirai for september" → args: {"detector": "moirai", "target_month": "september"}
- **ANOMALY WITH RULE**: If user says "find anomalies where values are > 2000" → args: {"detector": "rule", "custom_prompt": "values are > 2000"}

After receiving tool results, provide a clear, natural language summary that includes:
1. The key findings with specific numbers
2. Which models were used and why (from model_selection_rationale if available)
3. Confidence assessment based on model agreement
If no tool is needed (e.g., general questions about capabilities), respond directly.
"""


# Karpathy-style Council-of-AI: Expert roles for deliberation
COUNCIL_EXPERTS = {
    "statistician": {
        "name": "Dr. Stats",
        "role": "Senior Statistician",
        "emoji": "📊",
        "prompt": """You are Dr. Stats, a senior statistician with 20 years of experience in time series analysis.
You speak precisely and methodically. You focus on:
- Statistical significance and confidence intervals
- Distribution characteristics and normality tests
- Appropriate statistical methods and their assumptions
- Mathematical rigor in your assessments

Provide your COMPLETE analysis. Don't hold back - give your full expert opinion with specific numbers and statistical reasoning."""
    },

    "domain_expert": {
        "name": "Alex",
        "role": "Domain Expert",
        "emoji": "🎯",
        "prompt": """You are Alex, a domain expert who has worked with this type of data for 15 years.
You understand the real-world context behind the numbers. You focus on:
- What the patterns mean in practical terms
- Historical context and comparisons
- Potential causes behind the observed behavior
- Real-world implications of the findings

Provide your COMPLETE analysis. Draw on your domain expertise to explain what's really happening."""
    },

    "risk_analyst": {
        "name": "Morgan",
        "role": "Risk Analyst",
        "emoji": "⚠️",
        "prompt": """You are Morgan, a risk analyst who specializes in identifying threats and vulnerabilities.
You have a cautious, thorough approach. You focus on:
- Warning signs and red flags in the data
- Worst-case scenarios and their likelihood
- Volatility, uncertainty, and downside risks
- What could go wrong and how to mitigate it

Provide your COMPLETE risk assessment. Don't sugarcoat - highlight the real risks."""
    },

    "optimist": {
        "name": "Jordan",
        "role": "Opportunity Analyst",
        "emoji": "🚀",
        "prompt": """You are Jordan, an opportunity analyst who finds the silver linings.
You balance realism with optimism. You focus on:
- Positive trends and growth opportunities
- Upside potential and best-case scenarios
- Actionable recommendations for improvement
- How to capitalize on the patterns observed

Provide your COMPLETE opportunity analysis. Be constructive and forward-looking."""
    },

    "synthesizer": {
        "name": "Sam",
        "role": "Chief Analyst",
        "emoji": "🧠",
        "prompt": """You are Sam, the Chief Analyst who synthesizes all perspectives into actionable insights.
You are diplomatic but decisive. Your job is to:
- Weigh different viewpoints fairly
- Identify areas of agreement and disagreement
- Make clear recommendations based on the evidence
- Provide a balanced, nuanced final verdict

Create the FINAL SYNTHESIS. Integrate all perspectives into clear recommendations."""
    }
}

# Legacy council roles (kept for backward compatibility)
COUNCIL_ROLES = {
    "forecaster": COUNCIL_EXPERTS["statistician"]["prompt"],
    "risk_analyst": COUNCIL_EXPERTS["risk_analyst"]["prompt"],
    "business_explainer": COUNCIL_EXPERTS["optimist"]["prompt"],
}


SUGGESTION_PROMPT = """You are a helpful assistant suggesting follow-up actions for a time series analysis tool.

Based on the user's last question and the tool execution results, suggest exactly 3 short, relevant follow-up questions or commands.

Available Capabilities:
- Forecasting ("Forecast next 7 days", "Predict next month")
- Anomaly Detection ("Check for anomalies", "Any outliers in Oct?")
- Series Description ("Describe the trend", "Statistics")
- What-If Simulation ("What if value increases 10%?")
- Comparison ("Compare with last year", "Compare first half vs second half")
- Sensitivity Analysis ("Sensitivity of forecast")

Constraints:
1. ONLY suggest actions from the Available Capabilities list.
2. DO NOT suggest "Plot", "Graph", "Visualize" or specific chart customizations (the UI handles charts automatically).
3. Short and punchy (max 6-8 words).
4. Relevant to the context.

Example Output:
["Forecast next 7 days", "Check anomalies in May", "What if sales drop 10%?"]

Response must be a valid JSON list of strings.
"""


class Orchestrator:
    """
    Main orchestration loop for time series analysis.

    Flow:
    1. User asks a question
    2. LLM decides which tool to call
    3. Tool executes and returns results
    4. LLM summarizes results for user
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        csv_path: str = None,
        target_col: str = None,
        council_providers: Optional[Dict[str, BaseLLMProvider]] = None,
        progress_callback: Optional[Callable[[ProgressStage, str, float], None]] = None,
        use_skills: bool = True,
        forecaster: Optional[Any] = None,
        detector: Optional[Any] = None,
        data: Optional[Any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            llm_provider: LLM provider instance
            csv_path: Path to the CSV file being analyzed
            target_col: Name of the target column
            council_providers: Optional dict mapping council role names to providers
            progress_callback: Optional callback for progress updates
                              Signature: (stage, message, progress 0-1)
            use_skills: Whether to use the new skills architecture
            forecaster: Optional default forecaster instance
            detector: Optional default detector instance
            data: Pre-loaded pd.Series (with DatetimeIndex) or pd.DataFrame
                  (alternative to csv_path). When a Series is provided,
                  target_col is optional.
        """
        if csv_path is None and data is None:
            raise ValueError(
                "Either 'csv_path' or 'data' (pd.Series/pd.DataFrame) must be provided"
            )

        self.llm = llm_provider
        self.csv_path = csv_path
        self.target_col = target_col
        self._input_data = data  # Store raw input for _load_data
        self.conversation_history = []
        self.progress_callback = progress_callback
        self.use_skills = use_skills

        # Council providers: use provided or default to main provider
        self.council_providers = council_providers or {
            "forecaster": llm_provider,
            "risk_analyst": llm_provider,
            "business_explainer": llm_provider
        }

        # Initialize skills architecture
        if use_skills:
            dynamic_skills_enabled = os.getenv("TS_ENABLE_DYNAMIC_SKILLS", "false").strip().lower() in {
                "1", "true", "yes", "on"
            }

            load_skills()  # Load skill definitions
            self.skill_executor = SkillExecutor(
                llm_provider=llm_provider,
                forecasters={forecaster.name: forecaster} if forecaster else {},
                detectors={detector.name: detector} if detector else {},
            )
            if dynamic_skills_enabled:
                self.skill_generator = DynamicSkillGenerator(
                    llm_provider=llm_provider,
                )
            else:
                self.skill_generator = None
                logger.info("Dynamic skill generation disabled (TS_ENABLE_DYNAMIC_SKILLS=false)")
        else:
            self.skill_executor = None
            self.skill_generator = None

        # Detection memory (auto-accumulates across calls)
        self._detection_memory = None

        # Load data
        self._data = None
        self._data_context = None
        self._load_data()

        logger.info(f"Initialized orchestrator for {csv_path or 'in-memory data'}:{self.target_col}")
        logger.info(f"Using provider: {llm_provider.provider_name}")
        logger.info(f"Skills mode: {'enabled' if use_skills else 'disabled'}")

    def _load_data(self) -> None:
        """Load data from CSV file or use provided in-memory data."""
        try:
            if self._input_data is not None:
                # Use provided in-memory data
                if isinstance(self._input_data, pd.Series):
                    # Convert Series to DataFrame
                    s = self._input_data.copy()
                    if not isinstance(s.index, pd.DatetimeIndex):
                        s.index = pd.to_datetime(s.index)
                    col_name = s.name or self.target_col or "value"
                    self._data = pd.DataFrame({col_name: s})
                    if self.target_col is None:
                        self.target_col = col_name
                    # Store the series for direct injection into tools
                    self._series = s
                elif isinstance(self._input_data, pd.DataFrame):
                    self._data = self._input_data.copy()
                    if not isinstance(self._data.index, pd.DatetimeIndex):
                        self._data.index = pd.to_datetime(self._data.index)
                    # Infer target_col if not provided
                    if self.target_col is None:
                        numeric_cols = self._data.select_dtypes(include=["number"]).columns
                        if len(numeric_cols) > 0:
                            self.target_col = numeric_cols[0]
                        else:
                            self.target_col = self._data.columns[0]
                    self._series = self._data[self.target_col].dropna()
                else:
                    raise ValueError(
                        f"'data' must be pd.Series or pd.DataFrame, got {type(self._input_data)}"
                    )
                logger.info(f"Loaded in-memory data: {len(self._data)} rows")
            else:
                # Load from CSV
                from .utils import load_timeseries_csv
                self._data = load_timeseries_csv(self.csv_path)
                self._series = None  # Will use csv_path path in tools
                logger.info(f"Loaded CSV data: {len(self._data)} rows")

            self._data_context = DataContext(
                data=self._data,
                target_col=self.target_col,
                date_col=self._data.index.name,
                metadata={"csv_path": self.csv_path} if self.csv_path else {},
            )
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self._data = None
            self._data_context = None
            self._series = None

    def set_detection_memory(self, memory) -> None:
        """Set detection memory for context-aware detection.

        Args:
            memory: DetectionMemory instance with previous anomalies,
                    baseline stats, and domain context.
        """
        self._detection_memory = memory

    def _auto_update_memory(self, detection_result: Dict[str, Any]) -> None:
        """Auto-accumulate detection results into memory.

        Extracts baseline_stats (mean, std, median) from the detection result
        metadata so subsequent detection calls can use them as a reference
        baseline for scoring.
        """
        from .types import DetectionMemory

        if self._detection_memory is None:
            self._detection_memory = DetectionMemory()

        # Update baseline_stats from the detection result metadata
        metadata = detection_result.get("metadata", {})
        if isinstance(metadata, dict):
            stats_update = {}
            for key in ("mean", "std", "median"):
                if key in metadata and metadata[key] is not None:
                    stats_update[key] = float(metadata[key])
            if stats_update:
                self._detection_memory.baseline_stats.update(stats_update)

    def _report_progress(self, stage: ProgressStage, message: str, progress: float):
        """Report progress through callback if available."""
        if self.progress_callback:
            self.progress_callback(stage, message, min(1.0, max(0.0, progress)))
        logger.debug(f"Progress [{stage.value}]: {progress:.0%} - {message}")

    def _build_context(self) -> str:
        """Build context string with file info."""
        try:
            if self._data is not None:
                df = self._data
            else:
                from .utils import load_timeseries_csv
                df = load_timeseries_csv(self.csv_path)
            columns = list(df.columns)
            n_rows = len(df)
            date_range = f"{df.index[0]} to {df.index[-1]}"
            
            # Calculate data frequency
            frequency_str = "unknown"
            steps_per_day = None
            if len(df) > 1:
                time_diffs = pd.Series(df.index).diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    frequency_str = str(median_diff)
                    
                    # Calculate steps per common time units
                    if hasattr(median_diff, 'total_seconds'):
                        seconds = median_diff.total_seconds()
                        if seconds > 0:
                            steps_per_hour = 3600 / seconds
                            steps_per_day = 86400 / seconds
                            steps_per_week = steps_per_day * 7

            # Build context with frequency info
            source = self.csv_path if self.csv_path else "In-memory data"
            context = (
                f"**Current Data Context:**\n"
                f"- File: {source}\n"
                f"- Target Column: {self.target_col}\n"
                f"- Available Columns: {columns}\n"
                f"- Data Points: {n_rows}\n"
                f"- Date Range: {date_range}\n"
                f"- Data Frequency: {frequency_str}\n"
            )
            
            # Add step conversion guide if we have frequency info
            if steps_per_day:
                context += (
                    f"\n**IMPORTANT - Horizon Conversion:**\n"
                    f"The 'horizon' parameter is in STEPS (data points), not time units.\n"
                    f"For this data:\n"
                    f"- 1 hour ≈ {int(steps_per_hour)} steps\n"
                    f"- 1 day ≈ {int(steps_per_day)} steps\n"
                    f"- 1 week ≈ {int(steps_per_week)} steps\n"
                    f"When user asks for 'next 7 days', use horizon={int(steps_per_day * 7)}.\n"
                )
            
            return context
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return f"**Data Context:** Error loading file: {e}"

    def _truncate_for_summary(self, result: Dict[str, Any], max_array_size: int = 20) -> Dict[str, Any]:
        """
        Truncate large arrays in tool results to avoid token overflow when summarizing.
        
        Keeps first 5, middle 5, and last 5 values plus statistics.
        """
        import copy
        import numpy as np
        
        def truncate_array(arr):
            """Truncate array and return summary with samples."""
            if not isinstance(arr, list) or len(arr) <= max_array_size:
                return arr
            
            n = len(arr)
            # Get samples: first 5, middle 5, last 5
            first = arr[:5]
            middle_start = n // 2 - 2
            middle = arr[middle_start:middle_start + 5]
            last = arr[-5:]
            
            # Calculate statistics
            arr_np = np.array(arr)
            stats = {
                "total_points": n,
                "first_5": first,
                "middle_5": middle, 
                "last_5": last,
                "min": float(np.min(arr_np)),
                "max": float(np.max(arr_np)),
                "mean": float(np.mean(arr_np)),
                "std": float(np.std(arr_np)),
            }
            return stats
        
        def truncate_dict(d):
            """Recursively truncate arrays in dict."""
            if not isinstance(d, dict):
                return d
            
            result = {}
            for k, v in d.items():
                if isinstance(v, list) and len(v) > max_array_size:
                    # Check if it's a list of numbers
                    if all(isinstance(x, (int, float)) for x in v[:10]):
                        result[k] = truncate_array(v)
                    else:
                        result[k] = v[:max_array_size]  # Just truncate
                elif isinstance(v, dict):
                    result[k] = truncate_dict(v)
                else:
                    result[k] = v
            return result
        
        return truncate_dict(copy.deepcopy(result))

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return results."""
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})

        if tool_name not in TOOLS:
            logger.error(f"Unknown tool: {tool_name}")
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        # Inject data source: prefer in-memory series, fall back to csv_path
        if self._series is not None and "series" not in args:
            args["series"] = self._series
        if "csv_path" not in args and self.csv_path:
            args["csv_path"] = self.csv_path
        if "target_col" not in args and self.target_col:
            args["target_col"] = self.target_col

        # Inject LLM provider for tools that support LLM-driven model selection
        if tool_name in ("detect_anomalies", "run_forecast") and "provider" not in args:
            args["provider"] = self.llm

        # Inject detection memory for anomaly detection
        if tool_name == "detect_anomalies" and self._detection_memory is not None:
            if "memory" not in args:
                args["memory"] = self._detection_memory

        logger.info(f"Executing tool: {tool_name} with args: {args}")
        self._report_progress(ProgressStage.TOOL_EXECUTION, f"Running {tool_name}...", 0.5)

        try:
            tool_func = TOOLS[tool_name]["function"]
            result = tool_func(**args)

            if result.get("success"):
                logger.info(f"Tool {tool_name} succeeded")
                # Auto-accumulate detection memory
                if tool_name == "detect_anomalies":
                    self._auto_update_memory(result)
            else:
                logger.warning(f"Tool {tool_name} failed: {result.get('error')}")

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} exception: {e}")
            return {"success": False, "error": str(e)}

    def _council_summarize(
        self,
        tool_call: Dict[str, Any],
        tool_result: Dict[str, Any],
        user_question: str,
        context: str
    ) -> str:
        """
        Get multi-perspective summary from the Council of AI.
        """
        perspectives = {}

        # Extract model selection info
        model_rationale = tool_result.get("model_selection_rationale", "")
        models_used = tool_result.get("models_used", [])

        self._report_progress(ProgressStage.COUNCIL, "Gathering council perspectives...", 0.6)

        for i, (role_name, role_prompt) in enumerate(COUNCIL_ROLES.items()):
            council_prompt = (
                f"{role_prompt}\n\n"
                f"{context}\n\n"
                f"**User Question:** {user_question}\n\n"
                f"**Tool Called:** {tool_call.get('tool')}\n\n"
                f"**Models Used:** {', '.join(models_used) if models_used else 'N/A'}\n\n"
                f"**Tool Result (truncated):**\n```json\n{json.dumps(self._truncate_for_summary(tool_result), indent=2)}\n```\n\n"
                f"Provide a 2-3 sentence interpretation from your perspective. "
                f"Reference which models were used and what that means for reliability."
            )

            provider = self.council_providers.get(role_name, self.llm)
            progress = 0.6 + (0.1 * (i + 1))
            self._report_progress(
                ProgressStage.COUNCIL,
                f"Consulting {role_name}...",
                progress
            )

            try:
                response = provider.generate(council_prompt)
                perspectives[role_name] = response.strip()
                logger.debug(f"Council {role_name} responded")
            except Exception as e:
                logger.error(f"Council {role_name} failed: {e}")
                perspectives[role_name] = f"[Error: {e}]"

        self._report_progress(ProgressStage.SYNTHESIZING, "Synthesizing perspectives...", 0.9)

        # Synthesize all perspectives
        rationale_section = f"**Model Selection Rationale:**\n{model_rationale}\n\n" if model_rationale else ""

        synthesis_prompt = (
            f"{context}\n\n"
            f"**User Question:** {user_question}\n\n"
            f"**Models Used:** {', '.join(models_used) if models_used else 'N/A'}\n\n"
            f"{rationale_section}"
            f"**Tool Result (truncated):**\n```json\n{json.dumps(self._truncate_for_summary(tool_result), indent=2)}\n```\n\n"
            f"**Council Perspectives:**\n\n"
            f"📊 **Quantitative Analyst:** {perspectives.get('forecaster', 'N/A')}\n\n"
            f"⚠️ **Risk Analyst:** {perspectives.get('risk_analyst', 'N/A')}\n\n"
            f"💼 **Business Insights:** {perspectives.get('business_explainer', 'N/A')}\n\n"
            f"Now synthesize these perspectives into a comprehensive response that:\n"
            f"1. Summarizes key findings with specific numbers\n"
            f"2. Explains which models were used and why\n"
            f"3. Assesses confidence based on model agreement\n"
            f"Keep it conversational but data-driven."
        )

        try:
            result = self.llm.generate(synthesis_prompt, system_instruction=SYSTEM_PROMPT)
            self._report_progress(ProgressStage.COMPLETE, "Analysis complete", 1.0)
            return result
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return (
                f"**Council Analysis:**\n\n"
                f"📊 **Forecast View:** {perspectives.get('forecaster', 'N/A')}\n\n"
                f"⚠️ **Risk View:** {perspectives.get('risk_analyst', 'N/A')}\n\n"
                f"💼 **Business View:** {perspectives.get('business_explainer', 'N/A')}"
            )

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return response.

        Args:
            user_message: The user's question

        Returns:
            Assistant response (either tool result summary or direct answer)
        """
        logger.info(f"Processing chat message: {user_message[:100]}...")
        self._report_progress(ProgressStage.INITIALIZING, "Analyzing question...", 0.1)

        context = self._build_context()

        # Step 1: Ask LLM what to do
        prompt = f"{context}\n\n**User Question:** {user_message}"

        self._report_progress(ProgressStage.TOOL_SELECTION, "Selecting tool...", 0.2)

        try:
            llm_response = self.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return str(e)

        # Step 2: Check if LLM wants to call a tool
        tool_call = self.llm.parse_tool_call(llm_response)

        if tool_call:
            logger.info(f"Tool call detected: {tool_call.get('tool')}")
            self._report_progress(ProgressStage.TOOL_EXECUTION, f"Executing {tool_call.get('tool')}...", 0.4)

            tool_result = self._execute_tool(tool_call)

            self._report_progress(ProgressStage.SYNTHESIZING, "Generating summary...", 0.8)

            # Step 3: Ask LLM to summarize the results
            # Extract model selection info if available
            model_rationale = tool_result.get("model_selection_rationale", "")
            models_used = tool_result.get("models_used", [])

            rationale_section = f"**Model Selection Rationale:**\n{model_rationale}\n\n" if model_rationale else ""

            # Truncate large arrays to avoid token overflow
            truncated_result = self._truncate_for_summary(tool_result)

            summary_prompt = (
                f"{context}\n\n"
                f"**User Question:** {user_message}\n\n"
                f"**Tool Called:** {tool_call.get('tool')}\n\n"
                f"**Models Used:** {', '.join(models_used) if models_used else 'N/A'}\n\n"
                f"{rationale_section}"
                f"**Tool Result (truncated for summary):**\n```json\n{json.dumps(truncated_result, indent=2)}\n```\n\n"
                f"Provide a clear, helpful summary that:\n"
                f"1. Highlights key findings with specific numbers\n"
                f"2. Explains which models were used and why they were selected\n"
                f"3. Notes the confidence level based on model agreement\n"
                f"Be conversational but data-driven."
            )

            try:
                final_response = self.llm.generate(summary_prompt, system_instruction=SYSTEM_PROMPT)
                self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)
                return final_response
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                return f"Tool executed successfully but error generating summary: {e}\n\nRaw result: {tool_result}"
        else:
            logger.info("No tool call, returning direct response")
            self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)
            return llm_response

    def chat_with_council(self, user_message: str) -> str:
        """
        Process a user message with Council-of-AI pattern.
        """
        logger.info(f"Processing council chat: {user_message[:100]}...")
        self._report_progress(ProgressStage.INITIALIZING, "Analyzing question...", 0.1)

        context = self._build_context()
        prompt = f"{context}\n\n**User Question:** {user_message}"

        self._report_progress(ProgressStage.TOOL_SELECTION, "Selecting tool...", 0.2)

        try:
            llm_response = self.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return str(e)

        tool_call = self.llm.parse_tool_call(llm_response)

        if tool_call:
            logger.info(f"Tool call detected: {tool_call.get('tool')}")
            self._report_progress(ProgressStage.TOOL_EXECUTION, f"Executing {tool_call.get('tool')}...", 0.4)

            tool_result = self._execute_tool(tool_call)

            return self._council_summarize(tool_call, tool_result, user_message, context)
        else:
            self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)
            return llm_response

    def interactive_loop(self):
        """Run interactive chat loop."""
        print("=" * 60)
        print("  Time Series Council - Interactive Mode")
        print("  Type 'quit' to exit, 'help' for example questions")
        print("=" * 60)
        print(self._build_context())
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "help":
                print("\nExample questions you can ask:")
                print("  - What will sales be in the next 7 days?")
                print("  - Describe the sales trend")
                print("  - Are there any anomalies in the data?")
                print("  - Forecast the next 14 days")
                print()
                continue

            response = self.chat(user_input)
            print(f"\nAssistant: {response}\n")

    def chat_with_skills(
        self,
        user_message: str,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message using the skills architecture.

        Args:
            user_message: The user's question
            user_context: Optional additional context from user

        Returns:
            Dict with response, skill_result, and thinking
        """
        if not self.use_skills or not self.skill_executor:
            # Fall back to tool-based chat
            return {
                "response": self.chat(user_message),
                "skill_result": None,
                "thinking": None,
            }

        logger.info(f"Processing with skills: {user_message[:100]}...")
        self._report_progress(ProgressStage.INITIALIZING, "Analyzing question...", 0.1)

        # Update data context with user context
        if self._data_context and user_context:
            self._data_context.user_context = user_context

        # Try to match a skill
        self._report_progress(ProgressStage.TOOL_SELECTION, "Matching skill...", 0.2)
        skill = self.skill_executor.match_skill(user_message)

        if skill:
            logger.info(f"Matched skill: {skill.name}")
            self._report_progress(
                ProgressStage.TOOL_EXECUTION,
                f"Executing {skill.name}...",
                0.4
            )

            # Execute the skill
            result = self.skill_executor.execute_from_query(
                user_message,
                data_context=self._data_context,
            )

            self._report_progress(ProgressStage.SYNTHESIZING, "Generating response...", 0.8)

            # Build response with LLM
            if result.success:
                summary = self._summarize_skill_result(user_message, result)
            else:
                summary = f"I encountered an issue: {result.error}"

            self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)

            return {
                "response": summary,
                "skill_result": result.to_dict(),
                "thinking": result.thinking,
            }
        else:
            # Try dynamic skill generation
            if self.skill_generator and self.skill_generator.can_generate():
                logger.info("No skill match, attempting dynamic generation")
                self._report_progress(
                    ProgressStage.TOOL_EXECUTION,
                    "Generating custom analysis...",
                    0.4
                )

                result = self.skill_generator.generate_and_execute(
                    user_message,
                    self._data_context,
                )

                if result.success:
                    self._report_progress(ProgressStage.SYNTHESIZING, "Generating response...", 0.8)
                    summary = self._summarize_skill_result(user_message, result)
                    self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)

                    return {
                        "response": summary,
                        "skill_result": result.to_dict(),
                        "thinking": result.thinking,
                    }

            # Fall back to LLM direct response
            logger.info("No skill match, using direct LLM response")
            self._report_progress(ProgressStage.SYNTHESIZING, "Generating response...", 0.8)

            context = self._build_context()
            prompt = f"{context}\n\n**User Question:** {user_message}"

            try:
                response = self.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
                self._report_progress(ProgressStage.COMPLETE, "Complete", 1.0)
                return {
                    "response": response,
                    "skill_result": None,
                    "thinking": "No matching skill found, responded directly.",
                }
            except Exception as e:
                return {
                    "response": f"I encountered an error: {e}",
                    "skill_result": None,
                    "thinking": None,
                }

    def generate_suggestions(
        self,
        user_message: str,
        tool_call: Optional[Dict[str, Any]] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        tool_summary: str = "",
    ) -> List[str]:
        """
        Generate dynamic follow-up suggestions based on context.
        """
        if not tool_call:
            # Fallback for non-tool messages
            return ["Describe the data", "Check for anomalies", "Forecast next week"]

        try:
            # Truncate result for context
            truncated_result = self._truncate_for_summary(tool_result or {})

            # Filter out non-serializable objects (like provider) from args
            from dataclasses import asdict, is_dataclass
            serializable_args = {}
            for k, v in tool_call.get('args', {}).items():
                if hasattr(v, 'generate'):  # Skip LLM provider objects
                    continue
                if is_dataclass(v) and not isinstance(v, type):
                    serializable_args[k] = asdict(v)
                else:
                    serializable_args[k] = v

            prompt = (
                f"{SUGGESTION_PROMPT}\n\n"
                f"User Question: {user_message}\n"
                f"Tool Called: {tool_call.get('tool')}\n"
                f"Tool Arguments: {json.dumps(serializable_args, default=str)}\n"
                f"Tool Summary: {tool_summary}\n\n"
                f"Tool Result (excerpt): {json.dumps(truncated_result)[:1000]}..."
            )

            response = self.llm.generate(prompt)

            # Parse JSON response
            try:
                # Basic cleanup
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                suggestions = json.loads(cleaned)
                if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                    return suggestions[:3]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse suggestions JSON: {response}")

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")

        # Fallback if generation fails
        return []

    def _summarize_skill_result(
        self,
        user_question: str,
        result: SkillResult,
    ) -> str:
        """Generate a natural language summary of skill result."""
        context = self._build_context()

        # Extract model selection rationale if available
        model_rationale = result.data.get("model_selection_rationale", "") if result.data else ""
        models_used = result.models_used or []
        rationale_section = f"**Model Selection Rationale:**\n{model_rationale}\n\n" if model_rationale else ""

        summary_prompt = (
            f"{context}\n\n"
            f"**User Question:** {user_question}\n\n"
            f"**Skill Used:** {result.skill_name}\n\n"
            f"**Models Used:** {', '.join(models_used) if models_used else 'N/A'}\n\n"
            f"{rationale_section}"
            f"**Result:**\n```json\n{json.dumps(result.data, indent=2)}\n```\n\n"
            f"Provide a clear, helpful summary that:\n"
            f"1. Highlights key findings with specific numbers\n"
            f"2. Explains which models were used and why (from the rationale above)\n"
            f"3. Notes the confidence level based on model agreement\n"
            f"Be conversational but data-driven."
        )

        try:
            return self.llm.generate(summary_prompt, system_instruction=SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Analysis complete using {result.skill_name}. Results: {json.dumps(result.data, indent=2)}"

    def chat_with_skills_and_council(
        self,
        user_message: str,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message with skills and council perspectives.

        Combines skills architecture with multi-perspective analysis.
        """
        # First, execute with skills
        skill_response = self.chat_with_skills(user_message, user_context)

        if not skill_response.get("skill_result"):
            return skill_response

        # Get council perspectives on the skill result
        result = skill_response["skill_result"]
        context = self._build_context()

        perspectives = {}
        self._report_progress(ProgressStage.COUNCIL, "Gathering council perspectives...", 0.6)

        for i, (role_name, role_prompt) in enumerate(COUNCIL_ROLES.items()):
            council_prompt = (
                f"{role_prompt}\n\n"
                f"{context}\n\n"
                f"**User Question:** {user_message}\n\n"
                f"**Skill Used:** {result.get('skill_name')}\n\n"
                f"**Result:**\n```json\n{json.dumps(result.get('data', {}), indent=2)}\n```\n\n"
                f"Provide a 2-3 sentence interpretation from your perspective."
            )

            provider = self.council_providers.get(role_name, self.llm)
            progress = 0.6 + (0.1 * (i + 1))
            self._report_progress(ProgressStage.COUNCIL, f"Consulting {role_name}...", progress)

            try:
                response = provider.generate(council_prompt)
                perspectives[role_name] = response.strip()
            except Exception as e:
                perspectives[role_name] = f"[Error: {e}]"

        # Synthesize
        self._report_progress(ProgressStage.SYNTHESIZING, "Synthesizing perspectives...", 0.9)

        synthesis_prompt = (
            f"{context}\n\n"
            f"**User Question:** {user_message}\n\n"
            f"**Skill Result:**\n```json\n{json.dumps(result.get('data', {}), indent=2)}\n```\n\n"
            f"**Council Perspectives:**\n\n"
            f"📊 **Quantitative Analyst:** {perspectives.get('forecaster', 'N/A')}\n\n"
            f"⚠️ **Risk Analyst:** {perspectives.get('risk_analyst', 'N/A')}\n\n"
            f"💼 **Business Insights:** {perspectives.get('business_explainer', 'N/A')}\n\n"
            f"Synthesize these perspectives into a comprehensive response."
        )

        try:
            synthesized = self.llm.generate(synthesis_prompt, system_instruction=SYSTEM_PROMPT)
        except Exception as e:
            synthesized = skill_response["response"]

        self._report_progress(ProgressStage.COMPLETE, "Analysis complete", 1.0)

        return {
            "response": synthesized,
            "skill_result": result,
            "thinking": skill_response.get("thinking"),
            "perspectives": perspectives,
        }

    def karpathy_council(
        self,
        user_message: str,
        tool_result: Dict[str, Any],
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Karpathy-style Council: Full deliberation with multiple AI experts.

        This implements a multi-expert deliberation pattern where:
        1. Each expert analyzes the data from their perspective
        2. Experts see each other's analyses (round-table discussion)
        3. A synthesizer creates the final recommendation
        4. The entire deliberation process is returned

        Args:
            user_message: The user's question
            tool_result: The raw tool/analysis result to deliberate on
            user_context: Optional additional context

        Returns:
            Dict with full deliberation transcript and final synthesis
        """
        logger.info(f"Starting Karpathy Council deliberation...")
        context = self._build_context()

        # Deliberation transcript
        deliberation = {
            "question": user_message,
            "experts": [],
            "round_table": [],
            "synthesis": None,
            "full_transcript": ""
        }

        transcript_parts = [
            "=" * 60,
            "🏛️ COUNCIL OF AI EXPERTS - DELIBERATION SESSION",
            "=" * 60,
            f"\n📋 Question: {user_message}\n",
            "-" * 60,
        ]

        # Phase 1: Individual Expert Analyses
        expert_analyses = {}
        experts_to_consult = ["statistician", "domain_expert", "risk_analyst", "optimist"]

        self._report_progress(ProgressStage.COUNCIL_STAGE_1, "Gathering expert analyses...", 0.1)

        for i, expert_key in enumerate(experts_to_consult):
            expert = COUNCIL_EXPERTS[expert_key]
            progress = 0.1 + (0.15 * (i + 1))
            self._report_progress(
                ProgressStage.COUNCIL_STAGE_1,
                f"Consulting {expert['name']}...",
                progress
            )

            expert_prompt = (
                f"{expert['prompt']}\n\n"
                f"## Context\n{context}\n\n"
                f"## Question\n{user_message}\n\n"
                f"## Analysis Results (truncated)\n```json\n{json.dumps(self._truncate_for_summary(tool_result), indent=2)}\n```\n\n"
                f"{'## Additional Context: ' + user_context if user_context else ''}\n\n"
                f"Provide your complete expert analysis (4-6 paragraphs). Be specific with numbers and insights."
            )

            try:
                analysis = self.llm.generate(expert_prompt)
                expert_analyses[expert_key] = analysis.strip()

                deliberation["experts"].append({
                    "key": expert_key,
                    "name": expert["name"],
                    "role": expert["role"],
                    "emoji": expert["emoji"],
                    "analysis": analysis.strip()
                })

                transcript_parts.append(
                    f"\n{expert['emoji']} **{expert['name']}** ({expert['role']})\n"
                    f"{'-' * 40}\n"
                    f"{analysis.strip()}\n"
                )

            except Exception as e:
                logger.error(f"Expert {expert['name']} failed: {e}")
                expert_analyses[expert_key] = f"[Analysis unavailable: {e}]"

        # Phase 2: Round-Table Discussion (experts respond to each other)
        self._report_progress(ProgressStage.COUNCIL_STAGE_2, "Round-table discussion...", 0.7)

        transcript_parts.append("\n" + "=" * 60)
        transcript_parts.append("🗣️ ROUND-TABLE DISCUSSION")
        transcript_parts.append("=" * 60 + "\n")

        # Each expert comments on the others' analyses
        discussion_prompt = (
            f"You are participating in a round-table discussion. "
            f"Here are the perspectives shared by your colleagues:\n\n"
        )

        for expert_key, analysis in expert_analyses.items():
            expert = COUNCIL_EXPERTS[expert_key]
            discussion_prompt += f"**{expert['name']}**: {analysis[:500]}...\n\n"

        discussion_prompt += (
            f"\nAs a panel, identify:\n"
            f"1. Key points of agreement\n"
            f"2. Areas of disagreement or concern\n"
            f"3. Critical insights that should not be overlooked\n\n"
            f"Be concise but substantive."
        )

        try:
            discussion = self.llm.generate(discussion_prompt)
            deliberation["round_table"].append({
                "type": "panel_discussion",
                "content": discussion.strip()
            })
            transcript_parts.append(f"📢 **Panel Discussion**\n{discussion.strip()}\n")
        except Exception as e:
            logger.error(f"Round-table discussion failed: {e}")

        # Phase 3: Final Synthesis
        self._report_progress(ProgressStage.COUNCIL_STAGE_3, "Creating synthesis...", 0.85)

        transcript_parts.append("\n" + "=" * 60)
        transcript_parts.append("🧠 FINAL SYNTHESIS")
        transcript_parts.append("=" * 60 + "\n")

        synthesizer = COUNCIL_EXPERTS["synthesizer"]

        synthesis_prompt = (
            f"{synthesizer['prompt']}\n\n"
            f"## Original Question\n{user_message}\n\n"
            f"## Expert Analyses\n\n"
        )

        for expert_key, analysis in expert_analyses.items():
            expert = COUNCIL_EXPERTS[expert_key]
            synthesis_prompt += f"### {expert['name']} ({expert['role']})\n{analysis}\n\n"

        synthesis_prompt += (
            f"## Your Task\n"
            f"Create a comprehensive final synthesis that:\n"
            f"1. Summarizes the key findings all experts agreed on\n"
            f"2. Addresses any disagreements with a balanced view\n"
            f"3. Provides clear, actionable recommendations\n"
            f"4. Notes any caveats or areas needing further investigation\n\n"
            f"Be thorough but organized. Use headers and bullet points for clarity."
        )

        try:
            synthesis = self.llm.generate(synthesis_prompt)
            deliberation["synthesis"] = {
                "author": synthesizer["name"],
                "role": synthesizer["role"],
                "content": synthesis.strip()
            }
            transcript_parts.append(
                f"{synthesizer['emoji']} **{synthesizer['name']}** ({synthesizer['role']})\n"
                f"{synthesis.strip()}\n"
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            synthesis = "Synthesis generation failed."

        transcript_parts.append("\n" + "=" * 60)
        transcript_parts.append("END OF DELIBERATION")
        transcript_parts.append("=" * 60)

        deliberation["full_transcript"] = "\n".join(transcript_parts)

        self._report_progress(ProgressStage.COMPLETE, "Council deliberation complete", 1.0)

        return {
            "response": synthesis.strip() if isinstance(synthesis, str) else synthesis,
            "deliberation": deliberation,
            "tool_result": tool_result,
            "models_used": tool_result.get("models_used", []),
            "council_type": "single_llm",  # Fallback mode - single LLM with multiple personas
            "member_count": len(deliberation["experts"]),
        }

    def multi_llm_council(
        self,
        user_message: str,
        tool_result: Dict[str, Any],
        user_context: Optional[str] = None,
        providers: Optional[List[str]] = None,
        chairman: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        TRUE Karpathy-style Multi-LLM Council using different providers.

        This implements the real LLM Council pattern from https://github.com/karpathy/llm-council
        where multiple DIFFERENT LLM providers (Claude, GPT, Gemini, etc.) deliberate together.

        Three-stage process:
        1. Stage 1 - Independent Responses: Each LLM independently analyzes the question
        2. Stage 2 - Peer Review: Each LLM ranks other models' responses (anonymized)
        3. Stage 3 - Chairman Synthesis: A designated chairman synthesizes everything

        Args:
            user_message: The user's question
            tool_result: The raw tool/analysis result to deliberate on
            user_context: Optional additional context
            providers: List of provider names to use (defaults to all available)
            chairman: Provider name for the chairman (defaults to first available)

        Returns:
            Dict with full multi-LLM deliberation and final synthesis
        """
        from .council import MultiLLMCouncil

        logger.info("Starting TRUE Multi-LLM Council deliberation...")
        self._report_progress(ProgressStage.COUNCIL, "Initializing Multi-LLM Council...", 0.05)

        # Build context
        context = self._build_context()
        if user_context:
            context += f"\n\nUser Context: {user_context}"

        try:
            # Create the council with available providers
            council = MultiLLMCouncil(
                providers=providers,
                chairman=chairman,
                progress_callback=lambda msg, prog: self._report_progress(
                    ProgressStage.COUNCIL, msg, prog
                )
            )

            # Run the full deliberation
            deliberation = council.deliberate(
                question=user_message,
                context=context,
                tool_result=tool_result
            )

            # Convert to dict for JSON serialization
            deliberation_dict = council.to_dict(deliberation)

            self._report_progress(ProgressStage.COMPLETE, "Multi-LLM Council deliberation complete", 1.0)

            return {
                "response": deliberation.stage3_synthesis.final_response,
                "deliberation": deliberation_dict,
                "tool_result": tool_result,
                "models_used": [m.provider_name for m in council.members],
                "council_type": "multi_llm",
                "member_count": deliberation.member_count,
            }

        except Exception as e:
            logger.error(f"Multi-LLM Council failed: {e}")
            # Fall back to single-LLM council
            logger.info("Falling back to single-LLM council...")
            return self.karpathy_council(user_message, tool_result, user_context)
