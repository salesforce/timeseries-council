# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
API routes for Time Series Council web interface.
"""

import uuid
import asyncio
import tempfile
import os
import re
import time
from collections import deque
from functools import partial
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .models import (
    ChatRequest, ChatResponse, SessionConfig, SessionResponse,
    HealthResponse, ChatMode, ToolCall, CouncilPerspective,
    AdvancedCouncilResult, AdvancedCouncilStage1, AdvancedCouncilStage2,
    AdvancedCouncilStage3, AggregateRanking, ProgressUpdate, SkillExecution
)
from .progress import get_tracker, remove_tracker, ProgressTracker
from .chat_history import ChatHistoryManager
from .app import get_templates
from ..logging import get_logger
from ..types import ProgressStage

logger = get_logger(__name__)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with safe defaults."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_enabled_providers() -> set[str]:
    """Get comma-separated provider allowlist from env (empty = allow all)."""
    raw = os.getenv("TS_ENABLED_PROVIDERS", "").strip()
    if not raw:
        return set()  # empty means no restriction
    providers = {provider.strip().lower() for provider in raw.split(",") if provider.strip()}
    return providers


def _ensure_provider_allowed(provider_name: str) -> str:
    """Validate provider against deployment allowlist."""
    normalized = provider_name.lower().strip()
    enabled = _get_enabled_providers()
    if enabled and normalized not in enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{provider_name}' is disabled in this deployment. "
                f"Enabled providers: {', '.join(sorted(enabled))}"
            )
        )
    return normalized


def _is_raw_session_path_enabled() -> bool:
    """Whether direct file-path session creation is enabled."""
    return _parse_bool_env("TS_ENABLE_RAW_SESSION_PATH", default=False)


def _require_model_setup_access(request: Request) -> None:
    """Require admin token for model setup endpoints.

    If TS_ADMIN_TOKEN is unset, setup endpoints are disabled by default.
    """
    expected = os.getenv("TS_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="Model setup endpoints are disabled in this deployment."
        )

    provided = request.headers.get("X-Admin-Token", "").strip()
    if provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


def _new_id(length: int = 32) -> str:
    """Generate a strong opaque ID string."""
    value = uuid.uuid4().hex
    if length <= 0:
        return value
    return value[:min(length, len(value))]


def _sanitize_filename(filename: str) -> str:
    """Sanitize user-supplied filename for safe storage."""
    base_name = Path(filename or "").name
    if not base_name:
        return "upload.csv"

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)
    safe_name = safe_name.lstrip(".")
    return safe_name or "upload.csv"


def _get_max_upload_bytes() -> int:
    """Get max upload size in bytes from TS_MAX_UPLOAD_MB (default 20MB)."""
    raw_value = os.getenv("TS_MAX_UPLOAD_MB", "20")
    try:
        mb = int(raw_value)
    except ValueError:
        mb = 20
    mb = max(1, mb)
    return mb * 1024 * 1024


def _is_session_list_exposed() -> bool:
    """Whether /api/sessions endpoint is exposed."""
    return _parse_bool_env("TS_EXPOSE_SESSION_LIST", default=False)


def _require_session_api_access(request: Request) -> None:
    """Optional token gate for session/upload APIs.

    If TS_SESSION_API_TOKEN is unset, endpoint remains open.
    If set, clients must send X-Session-Token header.
    """
    expected = os.getenv("TS_SESSION_API_TOKEN", "").strip()
    if not expected:
        return

    provided = request.headers.get("X-Session-Token", "").strip()
    if provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


_rate_limit_hits: Dict[str, deque] = {}


def _is_rate_limit_enabled() -> bool:
    """Whether lightweight in-process rate limiting is enabled."""
    return _parse_bool_env("TS_RATE_LIMIT_ENABLED", default=True)


def _get_client_ip(request: Request) -> str:
    """Resolve client IP (supports reverse proxy forwarding headers)."""
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _get_rate_limit_config(bucket: str) -> tuple[int, int]:
    """Get (limit, window_seconds) for a rate-limit bucket."""
    default_limit = int(os.getenv("TS_RATE_LIMIT_DEFAULT_PER_WINDOW", "120"))
    window_seconds = int(os.getenv("TS_RATE_LIMIT_WINDOW_SECONDS", "60"))

    bucket_env_limits = {
        "upload": int(os.getenv("TS_RATE_LIMIT_UPLOAD_PER_WINDOW", "10")),
        "session": int(os.getenv("TS_RATE_LIMIT_SESSION_PER_WINDOW", "20")),
        "chat": int(os.getenv("TS_RATE_LIMIT_CHAT_PER_WINDOW", "60")),
        "admin": int(os.getenv("TS_RATE_LIMIT_ADMIN_PER_WINDOW", "10")),
    }

    return max(1, bucket_env_limits.get(bucket, default_limit)), max(1, window_seconds)


def _check_rate_limit(request: Request, bucket: str = "default") -> None:
    """Apply lightweight sliding-window rate limiting by IP and bucket."""
    if not _is_rate_limit_enabled():
        return

    ip = _get_client_ip(request)
    limit, window_seconds = _get_rate_limit_config(bucket)
    key = f"{bucket}:{ip}"

    now = time.time()
    hits = _rate_limit_hits.setdefault(key, deque())

    while hits and (now - hits[0]) > window_seconds:
        hits.popleft()

    if len(hits) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {bucket}. Try again later."
        )

    hits.append(now)


def _guard_session_api(request: Request, bucket: str = "default") -> None:
    """Apply session API token gate and per-IP rate limiting."""
    _require_session_api_access(request)
    _check_rate_limit(request, bucket)


def _make_json_safe(obj):
    """Convert values to JSON-safe types (handle NaN, Inf, numpy types)."""
    import numpy as np

    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if obj != obj or obj == float('inf') or obj == float('-inf'):  # NaN or Inf check
            return None
        return obj
    elif isinstance(obj, (np.ndarray,)):
        return _make_json_safe(obj.tolist())
    return obj


def _truncate_for_llm(result: Dict, max_array_size: int = 20) -> Dict:
    """
    Truncate large arrays in tool results to avoid token overflow when summarizing.
    Keeps first 5, middle 5, and last 5 values plus statistics.
    """
    import copy
    import numpy as np
    
    def truncate_array(arr):
        if not isinstance(arr, list) or len(arr) <= max_array_size:
            return arr
        
        n = len(arr)
        first = arr[:5]
        middle_start = max(0, n // 2 - 2)
        middle = arr[middle_start:middle_start + 5]
        last = arr[-5:]
        
        try:
            arr_np = np.array(arr)
            return {
                "total_points": n,
                "first_5": first,
                "middle_5": middle, 
                "last_5": last,
                "min": float(np.min(arr_np)),
                "max": float(np.max(arr_np)),
                "mean": float(np.mean(arr_np)),
                "std": float(np.std(arr_np)),
            }
        except:
            return {"total_points": n, "first_5": first, "last_5": last}
    
    def truncate_dict(d):
        if not isinstance(d, dict):
            return d
        
        result = {}
        for k, v in d.items():
            if isinstance(v, list) and len(v) > max_array_size:
                if v and all(isinstance(x, (int, float)) for x in v[:10]):
                    result[k] = truncate_array(v)
                else:
                    result[k] = v[:max_array_size]
            elif isinstance(v, dict):
                result[k] = truncate_dict(v)
            else:
                result[k] = v
        return result
    
    return truncate_dict(copy.deepcopy(result))


def _lttb_downsample(timestamps: List, values: List, target_points: int) -> tuple:
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.
    
    Preserves visual appearance of time series by keeping points that form
    the largest triangles with their neighbors - keeps peaks and troughs.
    
    Args:
        timestamps: List of timestamp values (any type)
        values: List of numeric values
        target_points: Desired number of output points
        
    Returns:
        Tuple of (downsampled_timestamps, downsampled_values)
    """
    n = len(values)
    
    if target_points >= n or target_points < 3:
        return timestamps, values
    
    # Convert to numeric indices for calculation
    sampled_indices = [0]  # Always keep first point
    
    # Bucket size (excluding first and last points)
    bucket_size = (n - 2) / (target_points - 2)
    
    a = 0  # Previous selected point index
    
    for i in range(target_points - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)
        
        # Calculate average of next bucket for reference
        next_bucket_start = int((i + 2) * bucket_size) + 1
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)
        
        if next_bucket_start < n:
            avg_x = (next_bucket_start + min(next_bucket_end, n)) / 2
            avg_y = sum(values[next_bucket_start:next_bucket_end]) / max(1, next_bucket_end - next_bucket_start)
        else:
            avg_x = n - 1
            avg_y = values[-1] if values else 0
        
        # Find point in current bucket that forms largest triangle
        max_area = -1
        max_idx = bucket_start
        
        for j in range(bucket_start, bucket_end):
            # Triangle area using the shoelace formula
            area = abs(
                (a - avg_x) * (values[j] - values[a]) -
                (a - j) * (avg_y - values[a])
            )
            if area > max_area:
                max_area = area
                max_idx = j
        
        sampled_indices.append(max_idx)
        a = max_idx
    
    sampled_indices.append(n - 1)  # Always keep last point
    
    # Extract sampled data
    sampled_timestamps = [timestamps[i] for i in sampled_indices]
    sampled_values = [values[i] for i in sampled_indices]
    
    return sampled_timestamps, sampled_values


def validate_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a CSV file for time series analysis.

    Checks:
    1. File can be parsed as CSV
    2. Has at least one datetime-like column (index)
    3. Has at least one numeric column for analysis
    4. Data is not malformed (values properly delimited)
    5. Has sufficient data points

    Returns:
        Dict with validation status and details
    """
    import pandas as pd
    import numpy as np

    result = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "columns": [],
        "numeric_columns": [],
        "datetime_columns": [],
        "row_count": 0,
        "date_range": None,
        "sample_data": None
    }

    try:
        # Try to read the CSV
        df = pd.read_csv(file_path)
        result["row_count"] = int(len(df))
        result["columns"] = list(df.columns)

        # Check for minimum rows
        if len(df) < 3:
            result["errors"].append(f"Insufficient data: only {len(df)} rows. Need at least 3 for analysis.")
            return result

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result["numeric_columns"] = numeric_cols

        if not numeric_cols:
            # Try to convert columns to numeric
            for col in df.columns:
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > len(df) * 0.5:  # At least 50% valid
                        numeric_cols.append(col)
                except:
                    pass
            result["numeric_columns"] = numeric_cols

        # Check for malformed data BEFORE numeric column filtering
        # This catches concatenated values that pandas reads as strings
        for col in df.columns:
            series = df[col]
            if series.dtype == object:
                for idx, val in enumerate(series.head(5)):
                    sample_val = str(val) if val is not None else ""
                    # Check for concatenated numbers (e.g., "CRM261.739990234375260.989990234375")
                    # Pattern: string with letters followed by long numeric sequence, or multiple decimals
                    if len(sample_val) > 15:
                        # Count decimal points - more than 2 suggests concatenated floats
                        decimal_count = sample_val.count('.')
                        # Check for letters mixed with long numeric sequences
                        import re
                        has_letter_then_numbers = re.match(r'^[A-Za-z]+[\d.]+$', sample_val)

                        if decimal_count > 2 or (has_letter_then_numbers and len(sample_val) > 20):
                            result["errors"].append(
                                f"Column '{col}' appears to have malformed data (values may be concatenated without proper delimiters). "
                                f"Sample value: '{sample_val[:60]}{'...' if len(sample_val) > 60 else ''}'"
                            )
                            return result

        if not numeric_cols:
            result["errors"].append("No numeric columns found. Time series analysis requires at least one numeric column.")
            return result

        # Check for datetime column
        datetime_cols = []
        first_col = df.columns[0]

        # Try first column as datetime
        try:
            df_with_dates = pd.read_csv(file_path, parse_dates=[0], index_col=0)
            if isinstance(df_with_dates.index, pd.DatetimeIndex):
                datetime_cols.append(first_col)
                result["date_range"] = {
                    "start": str(df_with_dates.index[0]),
                    "end": str(df_with_dates.index[-1])
                }
        except:
            pass

        # Also check other columns for datetime (but be more strict)
        for col in df.columns:
            if col not in datetime_cols and col not in numeric_cols:
                try:
                    # Only consider as datetime if values look like dates
                    sample = str(df[col].iloc[0])
                    # Skip if it looks numeric
                    if sample.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                        continue
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(df)
                    if valid_ratio > 0.8:
                        datetime_cols.append(col)
                except:
                    pass

        result["datetime_columns"] = datetime_cols

        if not datetime_cols:
            result["warnings"].append("No datetime column detected. First column will be used as index. For best results, include a datetime column.")

        # Check for data quality issues in numeric columns
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            series = df[col]

            # Check for NaN ratio
            nan_ratio = series.isna().sum() / len(series)
            if nan_ratio > 0.5:
                result["warnings"].append(f"Column '{col}' has {nan_ratio*100:.1f}% missing values.")

        # Check for non-numeric values in columns that should be numeric
        # This catches cases like "CRM" appearing in a price column
        for col in numeric_cols:
            if df[col].dtype == object:
                # Column is stored as strings - check for non-numeric values
                converted = pd.to_numeric(df[col], errors='coerce')
                non_numeric_mask = df[col].notna() & converted.isna()
                non_numeric_count = non_numeric_mask.sum()

                if non_numeric_count > 0:
                    # Get sample of non-numeric values
                    non_numeric_samples = df.loc[non_numeric_mask, col].head(3).tolist()
                    result["errors"].append(
                        f"Column '{col}' contains {non_numeric_count} non-numeric value(s) that need to be removed. "
                        f"Examples: {non_numeric_samples}"
                    )
                    result["needs_cleaning"] = True

        # Provide sample data (handle NaN values for JSON serialization)
        try:
            sample_df = df.head(3)
            sample_data = sample_df.to_dict(orient='records')
            result["sample_data"] = _make_json_safe(sample_data)
        except:
            pass

        # If we got here with no errors, it's valid
        if not result["errors"]:
            result["valid"] = True

    except pd.errors.EmptyDataError:
        result["errors"].append("CSV file is empty.")
    except pd.errors.ParserError as e:
        result["errors"].append(f"CSV parsing error: {str(e)}")
    except Exception as e:
        result["errors"].append(f"Error reading file: {str(e)}")

    return _make_json_safe(result)


def repair_csv_file(file_path: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to repair a malformed CSV file.

    Handles common issues:
    1. Concatenated values (e.g., "CRM261.73999260.98999" -> separate columns)
    2. Symbol prefix in numeric data
    3. Multiple floats concatenated without delimiters

    Args:
        file_path: Path to the CSV file
        validation_result: Previous validation result with error details

    Returns:
        Dict with repair status and new file path if successful
    """
    import pandas as pd
    import numpy as np
    import re
    from pathlib import Path

    result = {
        "repaired": False,
        "original_path": file_path,
        "repaired_path": None,
        "changes_made": [],
        "error": None
    }

    try:
        df = pd.read_csv(file_path)

        if len(df) < 1:
            result["error"] = "Empty file cannot be repaired"
            return result

        repaired_df = df.copy()
        changes = []

        # Check each column for malformed data
        for col in df.columns:
            series = df[col]

            if series.dtype != object:
                continue  # Skip already numeric columns

            # Sample values to detect pattern
            sample_vals = [str(v) for v in series.head(5).dropna()]
            if not sample_vals:
                continue

            sample = sample_vals[0]

            # Pattern 1: Symbol prefix + concatenated numbers (e.g., "CRM261.73999260.98999")
            symbol_match = re.match(r'^([A-Za-z]{1,5})([\d.]+)$', sample)
            if symbol_match and len(sample) > 15:
                symbol = symbol_match.group(1)
                numbers_str = symbol_match.group(2)

                # Try to split concatenated floats
                split_values = _split_concatenated_floats(numbers_str)

                if split_values and len(split_values) > 1:
                    # Determine column names based on count
                    if len(split_values) == 4:
                        new_cols = ['open', 'high', 'low', 'close']
                    elif len(split_values) == 5:
                        new_cols = ['open', 'high', 'low', 'close', 'volume']
                    elif len(split_values) == 6:
                        new_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
                    else:
                        new_cols = [f'value_{i+1}' for i in range(len(split_values))]

                    # Parse all rows
                    parsed_data = []
                    for val in series:
                        val_str = str(val) if val is not None else ""
                        m = re.match(r'^([A-Za-z]{1,5})([\d.]+)$', val_str)
                        if m:
                            nums = _split_concatenated_floats(m.group(2))
                            if nums and len(nums) == len(split_values):
                                parsed_data.append(nums)
                            else:
                                parsed_data.append([np.nan] * len(split_values))
                        else:
                            parsed_data.append([np.nan] * len(split_values))

                    # Add new columns
                    for i, new_col in enumerate(new_cols):
                        if new_col not in repaired_df.columns:
                            repaired_df[new_col] = [row[i] if i < len(row) else np.nan for row in parsed_data]

                    # Add symbol column if extracted
                    if 'symbol' not in repaired_df.columns:
                        repaired_df.insert(0, 'symbol', symbol)

                    # Remove original malformed column
                    repaired_df = repaired_df.drop(columns=[col])

                    changes.append(f"Split column '{col}' into: symbol, {', '.join(new_cols)}")
                    continue

            # Pattern 2: Multiple decimals (e.g., "261.73999260.98999" without symbol)
            decimal_count = sample.count('.')
            if decimal_count > 1 and len(sample) > 15:
                split_values = _split_concatenated_floats(sample)

                if split_values and len(split_values) > 1:
                    # Determine column names
                    if len(split_values) == 4:
                        new_cols = ['open', 'high', 'low', 'close']
                    elif len(split_values) == 5:
                        new_cols = ['open', 'high', 'low', 'close', 'volume']
                    else:
                        new_cols = [f'value_{i+1}' for i in range(len(split_values))]

                    # Parse all rows
                    parsed_data = []
                    for val in series:
                        val_str = str(val) if val is not None else ""
                        nums = _split_concatenated_floats(val_str)
                        if nums and len(nums) == len(split_values):
                            parsed_data.append(nums)
                        else:
                            parsed_data.append([np.nan] * len(split_values))

                    # Add new columns
                    for i, new_col in enumerate(new_cols):
                        if new_col not in repaired_df.columns:
                            repaired_df[new_col] = [row[i] if i < len(row) else np.nan for row in parsed_data]

                    # Remove original malformed column
                    repaired_df = repaired_df.drop(columns=[col])

                    changes.append(f"Split column '{col}' into: {', '.join(new_cols)}")

        # Pattern 3: Clean non-numeric values from columns that should be numeric
        # This handles cases like "CRM" appearing in a price column
        if validation_result.get("needs_cleaning"):
            for col in repaired_df.columns:
                if repaired_df[col].dtype == object:
                    # Try to convert to numeric
                    converted = pd.to_numeric(repaired_df[col], errors='coerce')
                    valid_ratio = converted.notna().sum() / len(repaired_df)

                    # If most values are numeric, this is likely a numeric column with some bad values
                    if valid_ratio > 0.5:
                        # Find rows with non-numeric values
                        non_numeric_mask = repaired_df[col].notna() & converted.isna()
                        non_numeric_count = non_numeric_mask.sum()

                        if non_numeric_count > 0:
                            # Remove rows with non-numeric values
                            rows_before = len(repaired_df)
                            repaired_df = repaired_df[~non_numeric_mask].copy()
                            rows_removed = rows_before - len(repaired_df)

                            # Convert column to numeric
                            repaired_df[col] = pd.to_numeric(repaired_df[col], errors='coerce')

                            changes.append(f"Removed {rows_removed} row(s) with non-numeric values from column '{col}'")

        # Also clean rows where datetime column is NaN/empty (often header rows or metadata)
        datetime_cols = validation_result.get("datetime_columns", [])
        if datetime_cols:
            first_dt_col = datetime_cols[0]
            if first_dt_col in repaired_df.columns:
                # Remove rows where datetime is NaN
                na_datetime_mask = repaired_df[first_dt_col].isna() | (repaired_df[first_dt_col].astype(str).str.strip() == '')
                if na_datetime_mask.any():
                    rows_before = len(repaired_df)
                    repaired_df = repaired_df[~na_datetime_mask].copy()
                    rows_removed = rows_before - len(repaired_df)
                    if rows_removed > 0:
                        changes.append(f"Removed {rows_removed} row(s) with missing datetime values")

        if changes:
            # Save repaired file
            repaired_path = Path(file_path).with_suffix('.repaired.csv')
            repaired_df.to_csv(repaired_path, index=False)

            result["repaired"] = True
            result["repaired_path"] = str(repaired_path)
            result["changes_made"] = changes
            result["new_columns"] = list(repaired_df.columns)
            result["row_count"] = int(len(repaired_df))  # Ensure JSON-serializable

            logger.info(f"Repaired CSV: {changes}")
        else:
            result["error"] = "No repairable patterns detected"

    except Exception as e:
        result["error"] = f"Repair failed: {str(e)}"
        logger.error(f"CSV repair error: {e}")

    return _make_json_safe(result)


def _split_concatenated_floats(s: str) -> list:
    """
    Split a string of concatenated float values.

    Strategy: Look for patterns where we have a float (digits.decimals) immediately
    followed by another float. The boundary is detected when decimal digits
    suddenly include another decimal point.

    Examples:
        "261.739990234375260.989990234375" -> [261.739990234375, 260.989990234375]
        "150.25151.30149.80150.90" -> [150.25, 151.30, 149.80, 150.90]

    Args:
        s: String with concatenated floats

    Returns:
        List of float values, or None if can't parse
    """
    import re

    if not s or not re.match(r'^[\d.]+$', s):
        return None

    # Count decimal points - need at least 2 for concatenation
    if s.count('.') < 2:
        return None

    values = []

    # Strategy: Find all positions of decimal points
    # Each float is: digits before decimal + decimal + digits after decimal
    # The tricky part is figuring out where one number's decimals end
    # and the next number's whole part begins

    # Use a smarter approach: split by looking for patterns like
    # "XXXXX.YYYYYY" where X are 1-4 digits and Y are decimal places
    # The key insight: stock prices are typically XXX.XX to XXXX.XXXXXX range

    # Try different decimal place lengths (12, 10, 8, 6, 4, 2)
    for decimal_places in [12, 10, 8, 6, 4, 2]:
        # Pattern: 1-5 digits, decimal, exactly N decimal digits
        pattern = rf'(\d{{1,5}}\.\d{{{decimal_places}}})'
        matches = re.findall(pattern, s)

        if matches and len(matches) >= 2:
            # Verify this pattern accounts for most of the string
            reconstructed = ''.join(matches)
            if len(reconstructed) >= len(s) * 0.9:  # At least 90% covered
                try:
                    values = [float(m) for m in matches]
                    return values
                except ValueError:
                    continue

    # Fallback: Try to split on boundaries where we see patterns like
    # "5260." - decimal digits followed by a new whole number
    # Look for where a single digit is followed by 2+ digits then a decimal

    # Another approach: Use regex to find overlapping patterns
    # Pattern: Look for decimal point, then find where next "whole number" starts
    all_floats = []

    # Find all decimal positions
    decimal_positions = [i for i, c in enumerate(s) if c == '.']

    if len(decimal_positions) < 2:
        return None

    # Each decimal marks a float. Work backwards from each decimal to find
    # where the whole number part starts (either beginning of string or
    # end of previous float's decimal part)
    try:
        current_pos = 0
        for i, dec_pos in enumerate(decimal_positions):
            # Find where this float starts
            start = current_pos

            # Find where the decimal part ends
            if i + 1 < len(decimal_positions):
                next_dec = decimal_positions[i + 1]
                # The decimal part ends somewhere before the next decimal
                # Look for where 2-4 digit whole number starts
                for end_pos in range(dec_pos + 2, next_dec):
                    # Check if remaining string to next decimal looks like start of new number
                    remaining_to_next = s[end_pos:next_dec]
                    if remaining_to_next and remaining_to_next.isdigit() and len(remaining_to_next) >= 2:
                        # This could be start of next number
                        float_str = s[start:end_pos]
                        if '.' in float_str:
                            try:
                                all_floats.append(float(float_str))
                                current_pos = end_pos
                                break
                            except ValueError:
                                continue
            else:
                # Last float - take rest of string
                float_str = s[start:]
                if '.' in float_str:
                    try:
                        all_floats.append(float(float_str))
                    except ValueError:
                        pass

        if len(all_floats) >= 2:
            return all_floats
    except Exception:
        pass

    return None


router = APIRouter()

# Session storage (in-memory for simplicity)
sessions: Dict[str, Dict[str, Any]] = {}

# File upload storage
uploaded_files: Dict[str, Dict[str, Any]] = {}


class UploadSessionConfig(BaseModel):
    """Config for session with uploaded file."""
    file_id: str
    target_col: str
    provider: str = "anthropic"
    mode: str = "council"
    user_context: Optional[str] = None


class SkillsSessionConfig(BaseModel):
    """Config for skills-based session."""
    csv_path: str
    target_col: str
    provider: str = "anthropic"
    mode: str = "council"
    user_context: Optional[str] = None
    use_skills: bool = True


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    templates = get_templates()
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    from ..providers import get_available_providers
    from ..forecasters import get_available_forecasters
    from ..detectors import get_available_detectors

    enabled = _get_enabled_providers()
    all_providers = get_available_providers()
    available = [p for p in all_providers if p in enabled] if enabled else all_providers

    return HealthResponse(
        status="ok",
        providers_available=available,
        forecasters_available=get_available_forecasters(),
        detectors_available=get_available_detectors()
    )


@router.get("/api/models/status")
async def get_model_status():
    """Get setup status for all models and detectors."""
    from ..setup_models import get_setup_status
    from ..forecasters import get_available_forecasters
    from ..detectors import get_available_detectors

    return {
        "setup_status": get_setup_status(),
        "available_forecasters": get_available_forecasters(),
        "available_detectors": get_available_detectors()
    }


@router.post("/api/models/setup/{model_name}")
async def setup_model_endpoint(request: Request, model_name: str, auto_install: bool = True):
    """
    Setup a specific model (install packages and download weights).

    Args:
        model_name: Name of the model/detector to setup
        auto_install: Whether to auto-install packages (default True)
    """
    _check_rate_limit(request, "admin")
    _require_model_setup_access(request)

    import asyncio
    from ..setup_models import setup_model

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: setup_model(model_name, auto_install=auto_install)
    )

    return result


@router.post("/api/models/setup-all")
async def setup_all_models(request: Request):
    """Setup all detectors and forecasters."""
    _check_rate_limit(request, "admin")
    _require_model_setup_access(request)

    import asyncio
    from ..setup_models import setup_all_detectors, setup_all_forecasters

    loop = asyncio.get_event_loop()

    detector_result = await loop.run_in_executor(None, setup_all_detectors)
    forecaster_result = await loop.run_in_executor(None, setup_all_forecasters)

    return {
        "detectors": detector_result,
        "forecasters": forecaster_result
    }


@router.post("/api/upload")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    """
    Upload one or more files for analysis.

    Returns a file_id that can be used to create a session.
    Validates CSV files to ensure they are in the correct format.
    """
    _guard_session_api(request, "upload")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_id = _new_id()
    saved_files = []
    validation_results = []
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json"}
    max_upload_bytes = _get_max_upload_bytes()

    # Create temp directory for this upload
    upload_dir = Path(tempfile.gettempdir()) / "timeseries_council" / file_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        original_name = file.filename or "upload"
        safe_name = _sanitize_filename(original_name)
        extension = Path(safe_name).suffix.lower()

        # Validate file type
        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {original_name}"
            )

        # Save file
        file_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        size = 0

        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_upload_bytes:
                    out.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File '{original_name}' exceeds upload limit "
                            f"of {max_upload_bytes // (1024 * 1024)} MB"
                        )
                    )
                out.write(chunk)

        file_info = {
            "name": original_name,
            "path": str(file_path),
            "size": size
        }

        # Validate CSV files
        if extension == ".csv":
            validation = validate_csv_file(str(file_path))

            # If validation failed, attempt auto-repair
            if not validation["valid"]:
                logger.warning(f"CSV validation issues for {original_name}: {validation['errors']}")
                logger.info(f"Attempting auto-repair for {original_name}...")

                repair_result = repair_csv_file(str(file_path), validation)

                if repair_result["repaired"]:
                    # Use repaired file
                    repaired_path = repair_result["repaired_path"]
                    file_info["path"] = repaired_path
                    file_info["original_path"] = str(file_path)
                    file_info["repaired"] = True
                    file_info["repair_changes"] = repair_result["changes_made"]

                    # Re-validate repaired file
                    validation = validate_csv_file(repaired_path)
                    logger.info(f"Repaired file validation: valid={validation['valid']}")

            file_info["validation"] = validation
            validation_results.append({
                "filename": original_name,
                "repaired": file_info.get("repaired", False),
                "repair_changes": file_info.get("repair_changes", []),
                **validation
            })

        saved_files.append(file_info)

    # Store file info with validation
    uploaded_files[file_id] = {
        "files": saved_files,
        "primary": saved_files[0]["path"] if saved_files else None,
        "upload_dir": str(upload_dir)
    }

    logger.info(f"Uploaded {len(saved_files)} files with id {file_id}")

    # Build response with validation info
    response = {
        "file_id": file_id,
        "files": [{"name": f["name"], "size": f["size"]} for f in saved_files],
        "primary_file": saved_files[0]["name"] if saved_files else None,
    }

    # Include validation results for CSV files
    if validation_results:
        response["validation"] = validation_results

        # Check repair status
        any_repaired = any(v.get("repaired", False) for v in validation_results)
        has_errors = any(not v["valid"] for v in validation_results)
        has_warnings = any(v.get("warnings") for v in validation_results)

        if has_errors:
            response["validation_status"] = "error"
            response["validation_message"] = "One or more files have validation issues. Check the 'validation' field for details."
        elif any_repaired:
            response["validation_status"] = "repaired"
            repair_details = [
                f"{v['filename']}: {', '.join(v.get('repair_changes', []))}"
                for v in validation_results if v.get("repaired")
            ]
            response["validation_message"] = f"Files were auto-repaired and are now valid. Changes: {'; '.join(repair_details)}"
        elif has_warnings:
            response["validation_status"] = "warning"
            response["validation_message"] = "Files uploaded with warnings. Check the 'validation' field for details."
        else:
            response["validation_status"] = "ok"
            response["validation_message"] = "All files validated successfully."

    return response


@router.get("/api/validate/{file_id}")
async def validate_uploaded_file(file_id: str, request: Request):
    """
    Validate an uploaded file without creating a session.

    Returns detailed validation results including:
    - Whether the file is valid for time series analysis
    - Available columns (numeric and datetime)
    - Any errors or warnings
    - Sample data preview
    """
    _guard_session_api(request, "session")

    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Upload not found.")

    file_info = uploaded_files[file_id]
    csv_path = file_info["primary"]

    if not csv_path:
        raise HTTPException(status_code=400, detail="No primary file in upload")

    if not csv_path.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Validation only supported for CSV files")

    validation = validate_csv_file(csv_path)

    return {
        "file_id": file_id,
        "filename": Path(csv_path).name,
        **validation
    }


@router.post("/api/session/upload", response_model=SessionResponse)
async def create_session_from_upload(config: UploadSessionConfig, request: Request):
    """
    Create a session using an uploaded file.
    Validates the file and target column before creating the session.
    """
    _guard_session_api(request, "session")

    config.provider = _ensure_provider_allowed(config.provider)

    if config.file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Upload not found. Please upload a file first.")

    file_info = uploaded_files[config.file_id]
    csv_path = file_info["primary"]

    if not csv_path:
        raise HTTPException(status_code=400, detail="No primary file in upload")

    # Validate CSV file before creating session
    if csv_path.endswith('.csv'):
        validation = validate_csv_file(csv_path)

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"CSV validation failed: {'; '.join(validation['errors'])}"
            )

        # Check if target column exists
        if config.target_col not in validation["columns"]:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{config.target_col}' not found. Available columns: {validation['columns']}"
            )

        # Warn if target column is not numeric
        if config.target_col not in validation["numeric_columns"]:
            logger.warning(f"Target column '{config.target_col}' may not be numeric")

    # Create session config
    session_config = SessionConfig(
        csv_path=csv_path,
        target_col=config.target_col,
        provider=config.provider
    )

    # Delegate to regular session creation
    from ..config import Config
    from ..orchestrator import Orchestrator

    session_id = _new_id()

    try:
        cfg = Config()
        provider = cfg.get_provider(config.provider)
        council_providers = cfg.get_council_providers()
        tracker = get_tracker(session_id)

        orchestrator = Orchestrator(
            llm_provider=provider,
            csv_path=csv_path,
            target_col=config.target_col,
            council_providers=council_providers,
            progress_callback=tracker.get_callback(),
            use_skills=True
        )

        # Create chat history manager
        chat_history = ChatHistoryManager(session_id)
        
        # Store session
        sessions[session_id] = {
            "orchestrator": orchestrator,
            "config": session_config,
            "provider_name": provider.provider_name,
            "tracker": tracker,
            "chat_history": chat_history,
            "file_id": config.file_id,
            "mode": config.mode,
            "user_context": config.user_context
        }

        file_name = Path(csv_path).name

        logger.info(f"Created session {session_id} from upload {config.file_id}")

        return SessionResponse(
            session_id=session_id,
            status="created",
            provider=provider.provider_name,
            csv_path=csv_path,
            target_col=config.target_col,
            file_name=file_name
        )

    except Exception as e:
        logger.error(f"Session creation from upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/skills")
async def list_skills():
    """List available skills."""
    from ..skills import get_registry, load_skills

    # Ensure skills are loaded
    load_skills()
    registry = get_registry()

    skills = registry.get_all()
    return {
        "skills": [skill.to_dict() for skill in skills],
        "count": len(skills)
    }


@router.post("/api/session", response_model=SessionResponse)
async def create_session(config: SessionConfig, request: Request):
    """
    Create a new chat session.
    """
    _guard_session_api(request, "session")

    if not _is_raw_session_path_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Direct file-path sessions are disabled in this deployment. "
                "Please use '/api/upload' followed by '/api/session/upload'."
            )
        )

    config.provider = _ensure_provider_allowed(config.provider)

    from ..config import Config
    from ..orchestrator import Orchestrator

    session_id = _new_id()

    try:
        cfg = Config()

        # Get main provider
        provider = cfg.get_provider(config.provider)

        # Get council providers
        council_providers = cfg.get_council_providers()

        # Create progress tracker
        tracker = get_tracker(session_id)

        # Create orchestrator with progress callback
        orchestrator = Orchestrator(
            llm_provider=provider,
            csv_path=config.csv_path,
            target_col=config.target_col,
            council_providers=council_providers,
            progress_callback=tracker.get_callback()
        )

        # Create chat history manager
        chat_history = ChatHistoryManager(session_id)
        
        # Store session
        sessions[session_id] = {
            "orchestrator": orchestrator,
            "config": config,
            "provider_name": provider.provider_name,
            "tracker": tracker,
            "chat_history": chat_history
        }

        logger.info(f"Created session {session_id} with provider {provider.provider_name}")

        return SessionResponse(
            session_id=session_id,
            status="created",
            provider=provider.provider_name,
            csv_path=config.csv_path,
            target_col=config.target_col,
            forecaster=config.forecaster,
            detector=config.detector
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"CSV file not found: {config.csv_path}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/progress/{session_id}")
async def stream_progress(session_id: str, request: Request):
    """
    Stream progress updates via Server-Sent Events.
    """
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    tracker = sessions[session_id].get("tracker")
    if not tracker:
        raise HTTPException(status_code=404, detail="No progress tracker for session")

    return StreamingResponse(
        tracker.stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/api/session/{session_id}/history")
async def get_chat_history(session_id: str, request: Request):
    """
    Retrieve chat history for a session.
    
    Returns all messages for UI display.
    """
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_history = sessions[session_id].get("chat_history")
    if not chat_history:
        return {"messages": []}
    
    messages = chat_history.get_all_messages()
    return {"messages": messages, "count": len(messages)}


@router.delete("/api/session/{session_id}/history")
async def clear_chat_history(session_id: str, request: Request):
    """Clear chat history for a session."""
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_history = sessions[session_id].get("chat_history")
    if chat_history:
        chat_history.clear_history()
    
    return {"status": "cleared", "session_id": session_id}


@router.get("/api/session/{session_id}/chart-data")
async def get_chart_data(
    request: Request,
    session_id: str,
    max_points: int = 500,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    resolution: str = "auto"
):
    """
    Get historical time series data for chart visualization.
    
    Supports dynamic resolution based on zoom level:
    - resolution="auto": Uses LTTB for overview, full for small ranges
    - resolution="full": Returns all points in range (use with caution)
    - resolution="lttb": Always use LTTB downsampling
    
    Args:
        session_id: Session identifier
        max_points: Maximum points to return (default 500)
        start_time: Optional ISO format start time for range query
        end_time: Optional ISO format end time for range query
        resolution: "auto", "full", or "lttb"
    """
    import pandas as pd

    _guard_session_api(request, "default")
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    orchestrator = session.get("orchestrator")
    
    if not orchestrator:
        raise HTTPException(status_code=400, detail="No orchestrator in session")
    
    try:
        # Read the CSV data
        csv_path = session["config"].csv_path
        target_col = session["config"].target_col
        
        df = pd.read_csv(csv_path)
        
        # Find datetime column for x-axis
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(5))
                    date_col = col
                    break
                except:
                    continue
        
        # Get values
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found")
        
        # Parse datetime column if found
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        
        # Apply time range filter if specified
        original_len = len(df)
        if start_time and end_time and date_col:
            try:
                start_dt = pd.to_datetime(start_time)
                end_dt = pd.to_datetime(end_time)
                df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
            except Exception as e:
                logger.warning(f"Could not parse time range: {e}")
        
        values = df[target_col].tolist()
        
        # Get timestamps
        if date_col:
            timestamps = df[date_col].astype(str).tolist()
        else:
            timestamps = list(range(len(values)))
        
        # Calculate data frequency (for frontend reference)
        frequency = None
        if date_col and len(df) > 1:
            time_diffs = df[date_col].diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                frequency = str(median_diff)
        
        # Determine if downsampling is needed
        is_downsampled = False
        downsampling_method = None
        
        if len(values) > max_points:
            if resolution == "full":
                # Return all points (frontend explicitly requested full resolution)
                pass
            elif resolution == "auto":
                # Auto: use LTTB for large datasets
                timestamps, values = _lttb_downsample(timestamps, values, max_points)
                is_downsampled = True
                downsampling_method = "lttb"
            else:  # "lttb"
                timestamps, values = _lttb_downsample(timestamps, values, max_points)
                is_downsampled = True
                downsampling_method = "lttb"
        
        # Get time range info
        time_range = None
        if date_col and len(df) > 0:
            time_range = {
                "start": str(df[date_col].min()),
                "end": str(df[date_col].max())
            }
        
        return _make_json_safe({
            "timestamps": timestamps,
            "values": values,
            "target_col": target_col,
            "date_col": date_col,
            "total_points": original_len,
            "displayed_points": len(values),
            "is_downsampled": is_downsampled,
            "downsampling_method": downsampling_method,
            "frequency": frequency,
            "time_range": time_range,
            "range_query": bool(start_time and end_time)
        })
        
    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/session/{session_id}/download")
async def download_results(
    request: Request,
    session_id: str,
    data_type: str,           # "historical", "forecast", "anomalies", "multi_model", "decomposition", "report"
    format: str = "auto",     # "auto", "csv", "json"
    include_flags: bool = True,       # For anomalies
    include_confidence: bool = True,  # For forecasts
):
    """
    Download analysis results in the specified format.
    
    Args:
        session_id: Session identifier
        data_type: Type of data to download
        format: Export format ("auto" uses original file format, or "csv"/"json")
        include_flags: Whether to include confidence flags for anomalies
        include_confidence: Whether to include confidence intervals for forecasts
        
    Returns:
        File download response
    """
    import pandas as pd
    from fastapi.responses import Response
    from .download import (
        format_forecast_csv, format_forecast_json,
        format_anomalies_csv, format_anomalies_json,
        format_multimodel_csv, format_multimodel_json,
        format_decomposition_csv, format_decomposition_json,
        format_historical_csv, format_historical_json,
        format_full_report_json, get_timestamp_str
    )
    
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Determine the original file format for "auto" mode
    csv_path = session["config"].csv_path
    original_format = "csv" if csv_path.lower().endswith(".csv") else "json"
    
    if format == "auto":
        format = original_format
    
    # Get historical data for context
    historical_data = None
    try:
        df = pd.read_csv(csv_path)
        target_col = session["config"].target_col
        
        # Find datetime column
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(5))
                    date_col = col
                    break
                except:
                    continue
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            timestamps = df[date_col].astype(str).tolist()
        else:
            timestamps = list(range(len(df)))
        
        historical_data = {
            "timestamps": timestamps,
            "values": df[target_col].tolist(),
            "target_col": target_col,
            "date_col": date_col
        }
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")
    
    # Get the last skill result from the session (if stored)
    # For now, we'll expect skill_result to be passed via query or stored in session
    skill_result = session.get("last_skill_result", {})
    skill_data = skill_result.get("data", {})
    
    timestamp_str = get_timestamp_str()
    
    try:
        if data_type == "historical":
            if format == "csv":
                content = format_historical_csv(historical_data or {})
                media_type = "text/csv"
                filename = f"historical_data_{timestamp_str}.csv"
            else:
                content = _make_json_safe(format_historical_json(historical_data or {}))
                content = __import__("json").dumps(content, indent=2)
                media_type = "application/json"
                filename = f"historical_data_{timestamp_str}.json"
        
        elif data_type == "forecast":
            if format == "csv":
                content = format_forecast_csv(skill_data, include_confidence, historical_data)
                media_type = "text/csv"
                filename = f"forecast_{timestamp_str}.csv"
            else:
                content = _make_json_safe(format_forecast_json(skill_data, include_confidence, historical_data))
                content = __import__("json").dumps(content, indent=2)
                media_type = "application/json"
                filename = f"forecast_{timestamp_str}.json"
        
        elif data_type == "anomalies":
            if format == "csv":
                content = format_anomalies_csv(skill_data, include_flags, historical_data)
                media_type = "text/csv"
                filename = f"anomalies_{timestamp_str}.csv"
            else:
                content = _make_json_safe(format_anomalies_json(skill_data, include_flags, historical_data))
                content = __import__("json").dumps(content, indent=2)
                media_type = "application/json"
                filename = f"anomalies_{timestamp_str}.json"
        
        elif data_type == "multi_model":
            if format == "csv":
                content = format_multimodel_csv(skill_result)
                media_type = "text/csv"
                filename = f"multi_model_comparison_{timestamp_str}.csv"
            else:
                content = _make_json_safe(format_multimodel_json(skill_result))
                content = __import__("json").dumps(content, indent=2)
                media_type = "application/json"
                filename = f"multi_model_comparison_{timestamp_str}.json"
        
        elif data_type == "decomposition":
            if format == "csv":
                content = format_decomposition_csv(skill_data)
                media_type = "text/csv"
                filename = f"decomposition_{timestamp_str}.csv"
            else:
                content = _make_json_safe(format_decomposition_json(skill_data))
                content = __import__("json").dumps(content, indent=2)
                media_type = "application/json"
                filename = f"decomposition_{timestamp_str}.json"
        
        elif data_type == "report":
            # Full report is always JSON
            session_info = {
                "session_id": session_id,
                "csv_path": csv_path,
                "target_col": session["config"].target_col,
                "provider": session.get("provider_name", "")
            }
            content = _make_json_safe(format_full_report_json(skill_result, historical_data, session_info))
            content = __import__("json").dumps(content, indent=2)
            media_type = "application/json"
            filename = f"full_report_{timestamp_str}.json"
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating download: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/session/{session_id}/store-result")
async def store_skill_result(session_id: str, request: Request):
    """
    Store the last skill result for download purposes.
    Called by frontend after receiving a skill result.
    """
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        body = await request.json()
        sessions[session_id]["last_skill_result"] = body.get("skill_result", {})
        return {"status": "stored"}
    except Exception as e:
        logger.error(f"Error storing skill result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/{session_id}", response_model=ChatResponse)
async def chat(session_id: str, request: Request, payload: ChatRequest):
    """
    Send a chat message and get a response.
    """
    _guard_session_api(request, "chat")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    orchestrator = session["orchestrator"]
    tracker = session.get("tracker")
    chat_history = session.get("chat_history")

    # Save user message to history
    if chat_history:
        chat_history.save_message("user", payload.message)

    # Get recent conversation context for LLM (last 3 exchanges)
    conversation_context = []
    if chat_history:
        conversation_context = chat_history.get_recent_context(last_n=3)

    # Reset progress tracker
    if tracker:
        tracker.reset()

    # Run chat in thread pool
    loop = asyncio.get_event_loop()

    try:
        if payload.mode == ChatMode.ADVANCED_COUNCIL:
            response, details = await loop.run_in_executor(
                None,
                partial(_chat_with_advanced_council, orchestrator, payload.message, conversation_context)
            )
            
            # Save assistant response
            if chat_history:
                chat_history.save_message("assistant", response)
            
            return ChatResponse(
                response=response,
                tool_call=ToolCall(**details["tool_call"]) if details.get("tool_call") else None,
                tool_result=details.get("tool_result"),
                advanced_council=details.get("advanced_council"),
                mode=payload.mode
            )
        elif payload.mode == ChatMode.COUNCIL:
            response, details = await loop.run_in_executor(
                None,
                partial(_chat_with_council, orchestrator, payload.message, conversation_context)
            )
        else:
            response, details = await loop.run_in_executor(
                None,
                partial(_chat_standard, orchestrator, payload.message, conversation_context)
            )

        # Build skill_result from details if present
        skill_result = None
        if details.get("skill_result"):
            skill_result = SkillExecution(
                skill_name=details["skill_result"].get("skill_name", "unknown"),
                success=details["skill_result"].get("success", False),
                data=details["skill_result"].get("data"),
                models_used=details["skill_result"].get("models_used"),
                execution_time=details["skill_result"].get("execution_time")
            )

        # Build deliberation for council mode
        deliberation = None
        if details.get("deliberation"):
            from .models import CouncilDeliberation, CouncilExpert
            delib_data = details["deliberation"]
            deliberation = CouncilDeliberation(
                experts=[
                    CouncilExpert(
                        key=e.get("key", ""),
                        name=e.get("name", ""),
                        role=e.get("role", ""),
                        emoji=e.get("emoji", ""),
                        analysis=e.get("analysis", "")
                    ) for e in delib_data.get("experts", [])
                ],
                round_table=delib_data.get("round_table", []),
                synthesis=delib_data.get("synthesis"),
                full_transcript=delib_data.get("full_transcript")
            )

        # Save assistant response to history with metadata
        if chat_history:
            metadata = {
                "skill_result": details.get("skill_result"),
                "thinking": details.get("thinking"),
                "models_used": details.get("skill_result", {}).get("models_used", []) if details.get("skill_result") else [],
                "suggestions": details.get("suggestions")
            }
            chat_history.save_message("assistant", response, metadata)

        return ChatResponse(
            response=response,
            tool_call=ToolCall(**details["tool_call"]) if details.get("tool_call") else None,
            tool_result=details.get("tool_result"),
            skill_result=skill_result,
            thinking=details.get("thinking"),
            council_perspectives=[
                CouncilPerspective(**p) for p in details.get("perspectives", [])
            ] if details.get("perspectives") else None,
            deliberation=deliberation,
            mode=payload.mode,
            suggestions=details.get("suggestions")
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        if tracker:
            tracker.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a chat session."""
    _guard_session_api(request, "default")

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    remove_tracker(session_id)
    del sessions[session_id]

    logger.info(f"Deleted session {session_id}")
    return {"status": "deleted", "session_id": session_id}


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """List all active sessions."""
    if not _is_session_list_exposed():
        raise HTTPException(status_code=404, detail="Not found")

    _guard_session_api(request, "default")

    return {
        "sessions": [
            {
                "session_id": sid,
                "provider": s["provider_name"],
                "csv_path": s["config"].csv_path
            }
            for sid, s in sessions.items()
        ]
    }


def _chat_standard(orchestrator, message: str, conversation_context: List[Dict[str, str]] = None) -> tuple:
    """Execute standard chat and extract tool call details."""
    import json
    import time
    from ..orchestrator import SYSTEM_PROMPT

    context = orchestrator._build_context()
    
    # Build conversation history string if available
    history_str = ""
    if conversation_context:
        history_str = "\n\n**Recent Conversation:**\n"
        for msg in conversation_context:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"
        history_str += "\n"
    
    prompt = f"{context}{history_str}\n**User Question:** {message}"

    llm_response = orchestrator.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
    tool_call = orchestrator.llm.parse_tool_call(llm_response)

    details = {}

    if tool_call:
        start_time = time.time()
        # Sanitize tool_call for response (remove non-serializable objects like provider)
        sanitized_tool_call = {
            "tool": tool_call.get("tool"),
            "args": {k: v for k, v in tool_call.get("args", {}).items() if not hasattr(v, 'generate')}
        }
        details["tool_call"] = sanitized_tool_call
        tool_result = orchestrator._execute_tool(tool_call)
        details["tool_result"] = tool_result
        execution_time = time.time() - start_time

        # Build skill result for UI
        tool_name = tool_call.get("tool", "unknown")
        details["skill_result"] = {
            "skill_name": tool_name,
            "success": tool_result.get("success", False),
            "data": tool_result,
            "models_used": [tool_result.get("model", tool_name)] if tool_result.get("model") else [tool_name],
            "execution_time": execution_time
        }

        # Build thinking info
        details["thinking"] = f"Selected tool: {tool_name}\nAnalyzing {orchestrator.target_col} from {orchestrator.csv_path}"

        # Truncate result to avoid token overflow
        truncated_result = _truncate_for_llm(tool_result)
        
        # Generate dynamic suggestions
        suggestions = orchestrator.generate_suggestions(
            user_message=message,
            tool_call=tool_call,
            tool_result=tool_result,
            tool_summary=tool_result.get("summary", "")
        )
        details["suggestions"] = suggestions
        
        summary_prompt = (
            f"{context}{history_str}\n"
            f"**User Question:** {message}\n\n"
            f"**Tool Called:** {tool_call.get('tool')}\n\n"
            f"**Tool Result (truncated for summary):**\n```json\n{json.dumps(truncated_result, indent=2)}\n```\n\n"
            f"Now provide a clear, helpful summary of these results for the user. "
            f"Mention which model/detector was used if relevant."
        )

        response = orchestrator.llm.generate(summary_prompt, system_instruction=SYSTEM_PROMPT)
    else:
        response = llm_response
        # Generate suggestions for direct response
        suggestions = orchestrator.generate_suggestions(user_message=message)
        details["suggestions"] = suggestions

    return response, details


def _chat_with_council(orchestrator, message: str, conversation_context: List[Dict[str, str]] = None) -> tuple:
    """Execute TRUE Karpathy-style Multi-LLM council chat with full deliberation."""
    import json
    import time
    from ..orchestrator import SYSTEM_PROMPT

    context = orchestrator._build_context()
    
    # Build conversation history string if available
    history_str = ""
    if conversation_context:
        history_str = "\n\n**Recent Conversation:**\n"
        for msg in conversation_context:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"
        history_str += "\n"
    
    prompt = f"{context}{history_str}\n**User Question:** {message}"

    llm_response = orchestrator.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
    tool_call = orchestrator.llm.parse_tool_call(llm_response)

    details = {}

    if tool_call:
        start_time = time.time()
        # Sanitize tool_call for response (remove non-serializable objects like provider)
        sanitized_tool_call = {
            "tool": tool_call.get("tool"),
            "args": {k: v for k, v in tool_call.get("args", {}).items() if not hasattr(v, 'generate')}
        }
        details["tool_call"] = sanitized_tool_call
        tool_result = orchestrator._execute_tool(tool_call)
        details["tool_result"] = tool_result

        # Build skill result for UI
        tool_name = tool_call.get("tool", "unknown")
        models_used = tool_result.get("models_used", [tool_name])

        details["skill_result"] = {
            "skill_name": tool_name,
            "success": tool_result.get("success", False),
            "data": tool_result,
            "models_used": models_used,
            "execution_time": None
        }

        # Build thinking info
        details["thinking"] = {
            "skill_selection": f"Selected tool: {tool_name}",
            "models_used": models_used,
            "model_rationale": tool_result.get("model_selection_rationale", "")
        }

        # Use TRUE Multi-LLM Council (multiple providers like Claude, GPT, Gemini)
        council_result = orchestrator.multi_llm_council(
            user_message=message,
            tool_result=tool_result
        )

        # Extract deliberation for UI
        deliberation = council_result.get("deliberation", {})
        council_type = council_result.get("council_type", "single_llm")

        # Build perspectives/experts for UI based on council type
        if council_type == "multi_llm":
            # Multi-LLM council has stage1 responses as "experts"
            perspectives = []
            for stage1 in deliberation.get("stage1", []):
                perspectives.append({
                    "role": stage1["member"],
                    "analysis": stage1["response"],
                    "emoji": next(
                        (m["emoji"] for m in deliberation.get("members", []) if m["name"] == stage1["member"]),
                        "🤖"
                    ),
                    "role_title": f"{stage1['provider']} ({stage1['model']})"
                })

            # Build experts list compatible with CouncilView
            experts = [
                {
                    "key": s["provider"],
                    "name": s["member"],
                    "role": f"{s['provider']} - {s['model']}",
                    "emoji": next(
                        (m["emoji"] for m in deliberation.get("members", []) if m["name"] == s["member"]),
                        "🤖"
                    ),
                    "analysis": s["response"]
                }
                for s in deliberation.get("stage1", [])
            ]

            # Build round_table from stage2 rankings
            round_table = []
            for ranking in deliberation.get("stage2", []):
                round_table.append({
                    "type": "ranking",
                    "member": ranking["member"],
                    "content": f"**Rankings:** {' > '.join(ranking['rankings'])}\n\n**Reasoning:** {ranking['reasoning']}"
                })

            # Build synthesis from stage3
            stage3 = deliberation.get("stage3", {})
            synthesis = {
                "author": stage3.get("chairman", "Chairman"),
                "role": "Chairman",
                "content": stage3.get("response", ""),
                "aggregate_rankings": stage3.get("aggregate_rankings", [])
            }

            details["deliberation"] = {
                "council_type": "multi_llm",
                "member_count": deliberation.get("member_count", len(experts)),
                "members": deliberation.get("members", []),
                "chairman": deliberation.get("chairman", {}),
                "full_transcript": deliberation.get("full_transcript", ""),
                "experts": experts,
                "round_table": round_table,
                "synthesis": synthesis,
                "stage1": deliberation.get("stage1", []),
                "stage2": deliberation.get("stage2", []),
                "stage3": stage3
            }
        else:
            # Fallback single-LLM council format
            perspectives = []
            for expert in deliberation.get("experts", []):
                perspectives.append({
                    "role": expert["name"],
                    "analysis": expert["analysis"],
                    "emoji": expert["emoji"],
                    "role_title": expert["role"]
                })

            details["deliberation"] = {
                "council_type": "single_llm",
                "member_count": len(deliberation.get("experts", [])),
                "full_transcript": deliberation.get("full_transcript", ""),
                "experts": deliberation.get("experts", []),
                "round_table": deliberation.get("round_table", []),
                "synthesis": deliberation.get("synthesis")
            }

        details["perspectives"] = perspectives
        response = council_result.get("response", "Council deliberation failed.")
        details["skill_result"]["execution_time"] = time.time() - start_time
    else:
        response = llm_response

    return response, details


def _chat_with_advanced_council(orchestrator, message: str, conversation_context: List[Dict[str, str]] = None) -> tuple:
    """Execute advanced council chat (Karpathy-style 3-stage deliberation)."""
    import json
    from ..orchestrator import SYSTEM_PROMPT
    from ..config import Config
    from ..council import AdvancedCouncil

    context = orchestrator._build_context()
    
    # Build conversation history string if available
    history_str = ""
    if conversation_context:
        history_str = "\n\n**Recent Conversation:**\n"
        for msg in conversation_context:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"
        history_str += "\n"
    
    prompt = f"{context}{history_str}\n**User Question:** {message}"

    llm_response = orchestrator.llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
    tool_call = orchestrator.llm.parse_tool_call(llm_response)

    details = {}

    if tool_call:
        # Sanitize tool_call for response (remove non-serializable objects like provider)
        sanitized_tool_call = {
            "tool": tool_call.get("tool"),
            "args": {k: v for k, v in tool_call.get("args", {}).items() if not hasattr(v, 'generate')}
        }
        details["tool_call"] = sanitized_tool_call
        tool_result = orchestrator._execute_tool(tool_call)
        details["tool_result"] = tool_result

        cfg = Config()
        adv_config = cfg.get_advanced_council_config()

        if adv_config["enabled"] and len(adv_config["providers"]) > 0:
            council = AdvancedCouncil(
                council_providers=adv_config["providers"],
                chairman_name=adv_config["chairman"]
            )

            council_query = (
                f"**Original Question:** {message}\n\n"
                f"**Data Context:**\n{context}\n\n"
                f"**Tool Used:** {tool_call.get('tool')}\n\n"
                f"**Tool Results (truncated):**\n```json\n{json.dumps(_truncate_for_llm(tool_result), indent=2)}\n```\n\n"
                f"Based on these results, provide your analysis and interpretation."
            )

            council_result = council.run_sync(council_query)

            details["advanced_council"] = AdvancedCouncilResult(
                stage1=[
                    AdvancedCouncilStage1(
                        model=r["model"],
                        response=r["response"],
                        provider=r.get("provider")
                    ) for r in council_result["stage1"]
                ],
                stage2=[
                    AdvancedCouncilStage2(
                        model=r["model"],
                        ranking=r["ranking"],
                        parsed_ranking=r.get("parsed_ranking"),
                        provider=r.get("provider")
                    ) for r in council_result["stage2"]
                ],
                stage3=AdvancedCouncilStage3(
                    model=council_result["stage3"]["model"],
                    response=council_result["stage3"]["response"],
                    is_fallback=council_result["stage3"].get("is_fallback", False)
                ),
                aggregate_rankings=[
                    AggregateRanking(
                        model=r["model"],
                        score=r.get("score", 0),
                        average_rank=r.get("average_rank", 0)
                    ) for r in council_result["metadata"].get("aggregate_rankings", [])
                ],
                chairman=council_result["metadata"].get("chairman")
            )

            response = council_result["stage3"]["response"]
        else:
            response = orchestrator.llm.generate(
                f"{context}\n\n**User Question:** {message}\n\n"
                f"**Tool Result (truncated):**\n```json\n{json.dumps(_truncate_for_llm(tool_result), indent=2)}\n```\n\n"
                f"Provide a clear analysis of these results.",
                system_instruction=SYSTEM_PROMPT
            )
    else:
        response = llm_response

    return response, details
