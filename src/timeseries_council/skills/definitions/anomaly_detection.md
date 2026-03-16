---
name: anomaly_detection
description: |
  Detect anomalies and outliers in time series data using multiple algorithms.
  Use when: user asks about anomalies, outliers, unusual patterns, or spikes.
triggers:
  - anomaly
  - anomalies
  - outlier
  - unusual
  - spike
  - abnormal
  - detect
parameters:
  - name: threshold
    type: float
    description: Sensitivity threshold (lower = more sensitive)
    default: 2.0
  - name: models
    type: array
    description: Specific detectors to use (empty for smart selection)
    default: []
  - name: context
    type: string
    description: Additional context about what to look for
    optional: true
requires_data: true
multi_model: true
---

# Anomaly Detection Skill

Identify anomalies, outliers, and unusual patterns in time series data using multiple detection algorithms.

## Model Selection Strategy

When `models` is empty, smart selection chooses 3-5 detectors based on:

### Data Characteristics
- **Normal distribution**: Z-Score, MAD
- **High volatility**: MAD, Isolation Forest
- **Unknown distribution**: Isolation Forest, LOF
- **Sparse anomalies**: LOF, DBSCAN

### Available Detectors
1. **zscore** - Statistical Z-score based detection
2. **mad** - Median Absolute Deviation (robust to outliers)
3. **isolation_forest** - Tree-based isolation approach
4. **lof** - Local Outlier Factor (density-based)

## Output

Returns:
- Indices of detected anomalies
- Anomaly scores for each point
- Threshold used for detection
- Cross-model comparison when multi_model is enabled

## Threshold Guidelines

- **1.5**: Very sensitive, many detections
- **2.0**: Standard sensitivity (default)
- **2.5**: Less sensitive, only strong anomalies
- **3.0**: Conservative, only extreme outliers

## Examples

**User**: "Are there any anomalies in this data?"
**Trigger**: "anomalies"
**Action**: Multi-model detection with default threshold

**User**: "Find unusual spikes in the sales data"
**Trigger**: "unusual", "spikes"
**Action**: Detection focused on upward anomalies

**User**: "Detect outliers with high sensitivity"
**Trigger**: "detect", "outliers"
**Action**: Detection with threshold=1.5
