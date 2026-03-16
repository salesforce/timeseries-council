---
name: simulation
description: |
  Run "what-if" scenarios to simulate changes in data (e.g. increase/decrease values).
  Use when: user asks "what if", "simulate increase", "amplify", "modify last N points".
triggers:
  - simulate
  - "what if"
  - scenario
  - amplify
  - increase
  - decrease
parameters:
  - name: scale_factor
    type: float
    description: "Multiplier for simulation (e.g. 1.2 = +20%, 0.8 = -20%)"
    default: 1.2
  - name: apply_to_last
    type: integer
    description: "Optional: Only apply simulation to the last N data points"
    default: null
  - name: horizon
    type: integer
    description: "Forecast horizon for projection"
    default: 14
requires_data: true
multi_model: false
---

# Simulation Skill

Run scenario-based simulations by scaling historical data.

## Parameters

- **scale_factor**: Multiplier to apply to the data.
    - `1.1` = 10% increase
    - `0.9` = 10% decrease
    - `2.0` = Double the values
- **apply_to_last**: If provided, only applies the scaling to the last N points of the series.
    - Useful for simulating recent shocks or changes.
    - Example: `apply_to_last: 30` simulates a change over the last month (approx).
- **horizon**: Number of future steps to project based on the simulated data.

## Examples

**User**: "Increase the last 30 data points by 10%"
**Trigger**: "increase"
**Action**: `what_if_simulation(scale_factor=1.1, apply_to_last=30)`

**User**: "What if the sales were 20% lower?"
**Trigger**: "what if"
**Action**: `what_if_simulation(scale_factor=0.8)`

**User**: "Simulate a shock where the last week was double"
**Trigger**: "simulate"
**Action**: `what_if_simulation(scale_factor=2.0, apply_to_last=7)`

