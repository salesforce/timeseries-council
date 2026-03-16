# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Structured logging for timeseries-council.

Default level: INFO
Configurable via:
- Environment variable: TIMESERIES_COUNCIL_LOG_LEVEL
- Programmatic: configure_logging(level="DEBUG")
- Config file: logging.level in config.yaml
"""

import logging
import os
import sys
from typing import Optional

# Package-wide logger name
LOGGER_NAME = "timeseries_council"

# Log formats
LOG_FORMAT_STANDARD = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_DETAILED = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)

# Track if logging has been configured
_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing data", extra={"rows": 1000})
    """
    # Ensure logging is configured
    if not _configured:
        configure_logging()

    # Handle both full paths and relative names
    if name.startswith(LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: Optional[str] = None,
    format_type: str = "standard",
    handler: Optional[logging.Handler] = None,
    force: bool = False
) -> None:
    """
    Configure logging for the entire package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to INFO or TIMESERIES_COUNCIL_LOG_LEVEL env var.
        format_type: 'standard' or 'detailed'
        handler: Custom handler (default: StreamHandler to stdout)
        force: Force reconfiguration even if already configured
    """
    global _configured

    if _configured and not force:
        return

    # Determine log level (priority: param > env > default)
    if level is None:
        level = os.environ.get("TIMESERIES_COUNCIL_LOG_LEVEL", "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create handler
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    # Set format
    fmt = LOG_FORMAT_STANDARD if format_type == "standard" else LOG_FORMAT_DETAILED
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(handler)
    logger.propagate = False

    _configured = True


def set_level(level: str) -> None:
    """
    Change the log level at runtime.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def disable_logging() -> None:
    """Disable all logging from this package."""
    logging.getLogger(LOGGER_NAME).disabled = True


def enable_logging() -> None:
    """Re-enable logging from this package."""
    logging.getLogger(LOGGER_NAME).disabled = False


# Initialize with defaults on import
configure_logging()
