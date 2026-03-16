# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Utility Functions for Time Series Council.
"""

from .data_loader import load_timeseries_csv, get_date_range, infer_frequency
from .date_parsing import (
    parse_month_reference,
    parse_month_name,
    filter_series_by_month,
    get_context_before_month,
    get_month_data_with_context,
    get_available_months,
    build_clarification_response,
    MONTH_NAMES,
    MONTH_NUMBER_TO_NAME,
)


def get_device(preferred: str = None) -> str:
    """
    Get the best available device for PyTorch operations.
    
    Args:
        preferred: If specified, use this device if available. 
                   Options: 'cuda', 'mps', 'cpu', or None for auto-detect.
    
    Returns:
        Device string: 'cuda' if NVIDIA GPU available, 'mps' if Apple Silicon,
                      otherwise 'cpu'.
    """
    # If user explicitly specified a device, respect it
    if preferred is not None:
        return preferred
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Check for Apple Silicon (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    return "cpu"


__all__ = [
    "load_timeseries_csv", 
    "get_date_range",
    "infer_frequency",
    "get_device",
    "parse_month_reference",
    "parse_month_name",
    "filter_series_by_month",
    "get_context_before_month",
    "get_month_data_with_context",
    "get_available_months",
    "build_clarification_response",
    "MONTH_NAMES",
    "MONTH_NUMBER_TO_NAME",
]
