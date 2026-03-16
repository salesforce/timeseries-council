# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Training Data Generator for Vertex AI Fine-tuning
==================================================
Generates synthetic dialogues for teaching the model to call the right tools.
"""
import json
import os
from pathlib import Path

# Training examples for each tool
# Format: (user_query, tool_name, args)

TRAINING_EXAMPLES = [
    # run_forecast examples (20)
    ("What will sales be next week?", "run_forecast", {"horizon": 7}),
    ("Predict the next 30 days", "run_forecast", {"horizon": 30}),
    ("Forecast revenue for the upcoming month", "run_forecast", {"horizon": 30}),
    ("What should I expect for tomorrow?", "run_forecast", {"horizon": 1}),
    ("Can you predict the next 2 weeks?", "run_forecast", {"horizon": 14}),
    ("What's the outlook for next quarter?", "run_forecast", {"horizon": 90}),
    ("Give me a 7 day forecast", "run_forecast", {"horizon": 7}),
    ("Project sales for the next 10 days", "run_forecast", {"horizon": 10}),
    ("What will happen next month?", "run_forecast", {"horizon": 30}),
    ("Estimate next week's numbers", "run_forecast", {"horizon": 7}),
    ("Show me predictions for 5 days ahead", "run_forecast", {"horizon": 5}),
    ("What's coming up in the next few days?", "run_forecast", {"horizon": 3}),
    ("Forecast the next year", "run_forecast", {"horizon": 365}),
    ("Predict tomorrow's value", "run_forecast", {"horizon": 1}),
    ("What will Q1 look like?", "run_forecast", {"horizon": 90}),
    ("Give me a long-term forecast", "run_forecast", {"horizon": 180}),
    ("Project the trend forward", "run_forecast", {"horizon": 14}),
    ("What are the expected values next week?", "run_forecast", {"horizon": 7}),
    ("Anticipate the next month's performance", "run_forecast", {"horizon": 30}),
    ("Model the future for 21 days", "run_forecast", {"horizon": 21}),
    
    # describe_series examples (15)
    ("Describe the sales data", "describe_series", {}),
    ("What's the trend over the last month?", "describe_series", {"window": 30}),
    ("Give me a summary of the data", "describe_series", {}),
    ("How has sales been performing?", "describe_series", {}),
    ("What's the overall picture?", "describe_series", {}),
    ("Tell me about this time series", "describe_series", {}),
    ("Summarize the recent data", "describe_series", {"window": 7}),
    ("What does the data look like?", "describe_series", {}),
    ("Give me the statistics", "describe_series", {}),
    ("How is the trend?", "describe_series", {}),
    ("Overview of the last 2 weeks", "describe_series", {"window": 14}),
    ("What's happening with the data?", "describe_series", {}),
    ("Describe the recent performance", "describe_series", {"window": 30}),
    ("What are the key metrics?", "describe_series", {}),
    ("Analyze the current state", "describe_series", {}),
    
    # detect_anomalies examples (15)
    ("Are there any unusual spikes in the data?", "detect_anomalies", {}),
    ("Find any outliers or anomalies", "detect_anomalies", {}),
    ("What's wrong with the data? Anything strange?", "detect_anomalies", {}),
    ("Detect anomalies with high sensitivity", "detect_anomalies", {"sensitivity": 1.5}),
    ("Look for any abnormal values", "detect_anomalies", {}),
    ("Are there any red flags?", "detect_anomalies", {}),
    ("Find suspicious data points", "detect_anomalies", {}),
    ("Check for unusual patterns", "detect_anomalies", {}),
    ("Identify any outliers", "detect_anomalies", {}),
    ("Any unexpected drops or spikes?", "detect_anomalies", {}),
    ("Scan for anomalies", "detect_anomalies", {}),
    ("What points look suspicious?", "detect_anomalies", {}),
    ("Find data quality issues", "detect_anomalies", {"sensitivity": 1.0}),
    ("Are there any extreme values?", "detect_anomalies", {}),
    ("Check for strange occurrences", "detect_anomalies", {}),
    
    # decompose_series examples (15)
    ("Break down the data into trend and seasonality", "decompose_series", {"period": 7}),
    ("What's the weekly pattern in sales?", "decompose_series", {"period": 7}),
    ("Decompose the series to see monthly patterns", "decompose_series", {"period": 30}),
    ("Show me the seasonal component", "decompose_series", {"period": 7}),
    ("Extract the trend from the noise", "decompose_series", {"period": 7}),
    ("What's the underlying pattern?", "decompose_series", {"period": 7}),
    ("Separate trend from seasonality", "decompose_series", {"period": 7}),
    ("Analyze the seasonal behavior", "decompose_series", {"period": 7}),
    ("What's the daily pattern?", "decompose_series", {"period": 1}),
    ("Show the monthly seasonality", "decompose_series", {"period": 30}),
    ("Decompose into components", "decompose_series", {"period": 7}),
    ("What's driving the pattern?", "decompose_series", {"period": 7}),
    ("Break it down by week", "decompose_series", {"period": 7}),
    ("Extract seasonal effects", "decompose_series", {"period": 7}),
    ("Show trend and residuals", "decompose_series", {"period": 7}),
    
    # compare_series examples (15)
    ("Compare all the metrics", "compare_series", {}),
    ("What's the correlation between sales and revenue?", "compare_series", {"columns": ["sales", "revenue"]}),
    ("How do different columns relate to each other?", "compare_series", {}),
    ("Show correlations between metrics", "compare_series", {}),
    ("Compare sales with marketing spend", "compare_series", {"columns": ["sales", "marketing_spend"]}),
    ("Which metrics are correlated?", "compare_series", {}),
    ("Analyze relationships between columns", "compare_series", {}),
    ("How does revenue correlate with visitors?", "compare_series", {"columns": ["revenue", "visitors"]}),
    ("Cross-compare all the data", "compare_series", {}),
    ("Show how metrics move together", "compare_series", {}),
    ("Find correlations in the data", "compare_series", {}),
    ("Compare revenue and sales trends", "compare_series", {"columns": ["revenue", "sales"]}),
    ("What variables are related?", "compare_series", {}),
    ("Correlation analysis please", "compare_series", {}),
    ("How do these metrics interact?", "compare_series", {}),
    
    # what_if_simulation examples (20)
    ("What if sales increase by 20%?", "what_if_simulation", {"scale_factor": 1.2}),
    ("Simulate a 30% drop in sales", "what_if_simulation", {"scale_factor": 0.7}),
    ("What would happen if we doubled our numbers?", "what_if_simulation", {"scale_factor": 2.0}),
    ("Impact of 15% growth over 30 days", "what_if_simulation", {"scale_factor": 1.15, "horizon": 30}),
    ("What if there's a 10% decline?", "what_if_simulation", {"scale_factor": 0.9}),
    ("Simulate 50% growth", "what_if_simulation", {"scale_factor": 1.5}),
    ("What happens with 25% reduction?", "what_if_simulation", {"scale_factor": 0.75}),
    ("Model a 5% increase", "what_if_simulation", {"scale_factor": 1.05}),
    ("What if sales tripled?", "what_if_simulation", {"scale_factor": 3.0}),
    ("Scenario with 40% drop", "what_if_simulation", {"scale_factor": 0.6}),
    ("Impact of halving the numbers", "what_if_simulation", {"scale_factor": 0.5}),
    ("What if we grow by 35%?", "what_if_simulation", {"scale_factor": 1.35}),
    ("Simulate worst case 50% decline", "what_if_simulation", {"scale_factor": 0.5}),
    ("Best case 100% growth scenario", "what_if_simulation", {"scale_factor": 2.0}),
    ("What if there's 8% growth?", "what_if_simulation", {"scale_factor": 1.08}),
    ("Model a slight 3% decrease", "what_if_simulation", {"scale_factor": 0.97}),
    ("What if performance drops by a quarter?", "what_if_simulation", {"scale_factor": 0.75}),
    ("Simulate moderate 12% growth", "what_if_simulation", {"scale_factor": 1.12}),
    ("Impact of 60% increase over 2 weeks", "what_if_simulation", {"scale_factor": 1.6, "horizon": 14}),
    ("What if we lose 20% of sales?", "what_if_simulation", {"scale_factor": 0.8}),
]


def generate_vertex_ai_format(examples, output_file="training_data.jsonl"):
    """
    Generate training data in Vertex AI JSONL format.
    
    Format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}
    """
    with open(output_file, 'w') as f:
        for query, tool, args in examples:
            tool_call = json.dumps({"tool": tool, "args": args})
            entry = {
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "model", "content": f"```json\n{tool_call}\n```"}
                ]
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Generated {len(examples)} training examples to {output_file}")
    return output_file


def estimate_cost(examples, epochs=3):
    """
    Estimate Vertex AI tuning cost.
    
    Based on ~$21/hour compute cost.
    Small dataset (100 examples) typically takes 15-30 min.
    """
    n_examples = len(examples)
    
    # Rough estimates
    if n_examples <= 50:
        estimated_time_hours = 0.25  # 15 min
    elif n_examples <= 100:
        estimated_time_hours = 0.5   # 30 min
    elif n_examples <= 200:
        estimated_time_hours = 0.75  # 45 min
    else:
        estimated_time_hours = 1.0   # 1 hour
    
    # Factor in epochs
    estimated_time_hours *= (epochs / 3)
    
    cost_per_hour = 21  # Approximate
    estimated_cost = estimated_time_hours * cost_per_hour
    
    print(f"\n📊 Cost Estimate:")
    print(f"   Examples: {n_examples}")
    print(f"   Epochs: {epochs}")
    print(f"   Estimated Time: {estimated_time_hours * 60:.0f} minutes")
    print(f"   Estimated Cost: ${estimated_cost:.2f}")
    
    return estimated_cost


if __name__ == "__main__":
    # Generate training data
    output_file = generate_vertex_ai_format(TRAINING_EXAMPLES)
    
    # Estimate cost
    estimate_cost(TRAINING_EXAMPLES, epochs=3)
    
    print(f"\n📁 Training data saved to: {output_file}")
    print(f"   Total examples: {len(TRAINING_EXAMPLES)}")
