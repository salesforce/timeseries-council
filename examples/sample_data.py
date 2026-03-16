# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Sample Data Generator for Time Series Council
==============================================
Creates synthetic time series data for testing.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_sales(
    n_days: int = 365,
    start_date: str = "2024-01-01",
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic daily sales data with trend, seasonality, and noise.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    
    # Base level
    base = 1000
    
    # Trend (slight growth)
    trend = np.linspace(0, 200, n_days)
    
    # Weekly seasonality (weekends lower)
    weekly = np.array([100 if d.weekday() < 5 else -150 for d in dates])
    
    # Monthly seasonality
    monthly = 50 * np.sin(2 * np.pi * np.arange(n_days) / 30)
    
    # Noise
    noise = np.random.normal(0, 50, n_days)
    
    # Combine
    sales = base + trend + weekly + monthly + noise
    sales = np.maximum(sales, 0)  # No negative sales
    
    # Add a few anomalies
    anomaly_idx = [50, 120, 250]
    for idx in anomaly_idx:
        if idx < n_days:
            sales[idx] *= 1.5  # Spike
    
    df = pd.DataFrame({
        "date": dates,
        "sales": sales.round(2)
    })
    df.set_index("date", inplace=True)
    
    if output_path:
        df.to_csv(output_path)
        print(f"Saved synthetic data to: {output_path}")
    
    return df


def generate_multi_metric_data(
    n_days: int = 365,
    start_date: str = "2024-01-01",
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic multi-metric data with correlated columns.
    Useful for testing compare_series tool.
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    
    # Base metrics
    base = 1000
    trend = np.linspace(0, 200, n_days)
    weekly = np.array([100 if d.weekday() < 5 else -150 for d in dates])
    noise = np.random.normal(0, 50, n_days)
    
    # Sales - primary metric
    sales = base + trend + weekly + noise
    sales = np.maximum(sales, 0)
    
    # Revenue - highly correlated with sales (price * sales)
    avg_price = 25 + np.random.normal(0, 2, n_days)
    revenue = sales * avg_price
    
    # Visitors - moderately correlated with sales
    conversion_rate = 0.15 + np.random.normal(0, 0.02, n_days)
    visitors = sales / np.clip(conversion_rate, 0.05, 0.3)
    
    # Marketing spend - slightly lagged correlation with sales
    marketing = 500 + trend * 0.5 + np.random.normal(0, 100, n_days)
    marketing = np.maximum(marketing, 0)
    
    df = pd.DataFrame({
        "date": dates,
        "sales": sales.round(2),
        "revenue": revenue.round(2),
        "visitors": visitors.round(0),
        "marketing_spend": marketing.round(2)
    })
    df.set_index("date", inplace=True)
    
    if output_path:
        df.to_csv(output_path)
        print(f"Saved multi-metric data to: {output_path}")
    
    return df


def generate_stock_like(
    n_days: int = 252,
    start_date: str = "2024-01-01",
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate random walk stock-like data.
    """
    np.random.seed(123)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")  # Business days
    
    # Random walk
    returns = np.random.normal(0.0005, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "date": dates,
        "price": price.round(2)
    })
    df.set_index("date", inplace=True)
    
    if output_path:
        df.to_csv(output_path)
        print(f"Saved stock data to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    generate_synthetic_sales(output_path=str(data_dir / "sample_sales.csv"))
    generate_stock_like(output_path=str(data_dir / "sample_stock.csv"))
    generate_multi_metric_data(output_path=str(data_dir / "sample_multi_metric.csv"))
    
    print("\nSample data files created!")
