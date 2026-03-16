---
name: analysis
description: |
  Perform statistical analysis on time series data.
  Use when: user asks about statistics, summary, describe data, or analyze.
triggers:
  - analyze
  - analysis
  - statistics
  - summary
  - describe
  - stats
  - mean
  - average
parameters:
  - name: include_trend
    type: boolean
    description: Include trend analysis
    default: true
  - name: include_seasonality
    type: boolean
    description: Include seasonality detection
    default: true
requires_data: true
multi_model: false
---

# Analysis Skill

Perform comprehensive statistical analysis on time series data.

## Analysis Components

### Basic Statistics
- Count, mean, standard deviation
- Min, max, median
- Quartiles (Q1, Q3)

### Trend Analysis
- Linear trend direction (increasing/decreasing)
- Trend slope magnitude
- Trend strength

### Distribution Analysis
- Skewness and kurtosis
- Normality assessment

## Output

Returns a comprehensive analysis object with:
- Descriptive statistics
- Trend information
- Data quality indicators

## Examples

**User**: "What's the average of this data?"
**Trigger**: "average"
**Action**: Basic statistics with mean highlighted

**User**: "Analyze the sales trends"
**Trigger**: "analyze", "trends"
**Action**: Full analysis with trend focus

**User**: "Give me a summary of the data"
**Trigger**: "summary"
**Action**: Comprehensive statistical summary
