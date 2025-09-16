"""
Lightweight logging utilities for excog-trajectory.

Provides a package-wide logger with level configurable via the EXCOG_LOG_LEVEL
environment variable. Defaults to INFO. Uses a simple, structured format.
"""
from __future__ import annotations

import logging
import os
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once with a sensible format.

    If already configured, this is a no-op. This function is safe to call
    multiple times.
    """
    if logging.getLogger().handlers:
        return
    level_value: int
    if isinstance(level, str) and level:
        level_value = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        level_value = level
    else:
        level_value = getattr(logging, os.getenv("EXCOG_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level_value, format=_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for module `name`."""
    configure_logging(None)
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
