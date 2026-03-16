---
name: forecasting
description: |
  Predict future values in time series data using AI models.
  Use when: user asks about predictions, forecasts, future values, or trends.
triggers:
  - forecast
  - predict
  - "what will"
  - future
  - projection
  - trend
parameters:
  - name: horizon
    type: integer
    description: Number of time steps to forecast
    default: 7
  - name: models
    type: array
    description: Specific models to use (empty for smart selection)
    default: []
  - name: context
    type: string
    description: Additional context about the forecast
    optional: true
requires_data: true
multi_model: true
---

# Forecasting Skill

Generate predictions for future time series values using multiple AI forecasting models.

## Model Selection Strategy

When `models` is empty, smart selection chooses 3-5 models based on:

### Data Characteristics
- **Short series (<100 points)**: Prefer simpler models (baseline, Moirai)
- **Seasonal data**: Chronos, TimesFM, Moirai
- **Financial/volatile data**: Lag-Llama, Chronos
- **Long series (>1000 points)**: TimesFM, Chronos

### Available Models
1. **moirai** - Salesforce's universal forecaster, good for general use
2. **chronos2** - Amazon's Chronos foundation model, strong seasonality detection with multivariate support
3. **timesfm** - Google's foundation model, handles long sequences
4. **lag_llama** - LLM-based forecaster, good for complex patterns
5. **zscore_baseline** - Simple statistical baseline

## Output

Returns predictions with:
- Point forecasts for each horizon step
- Confidence intervals (upper/lower bounds)
- Comparison across models when multi_model is enabled

## Examples

**User**: "What will sales look like next week?"
**Trigger**: "what will", "next"
**Action**: 7-day forecast with smart model selection

**User**: "Predict the next 30 days of temperature"
**Trigger**: "predict", "next"
**Action**: 30-day forecast with seasonal models
