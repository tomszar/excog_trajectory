"""
Custom exceptions for excog-trajectory.

Provide a small hierarchy for clearer error handling in user-facing code.
"""
from __future__ import annotations


class ExcogError(Exception):
    """Base exception for excog-trajectory."""


class ConfigurationError(ExcogError):
    """Raised when configuration is invalid or missing required values."""


class InputValidationError(ExcogError):
    """Raised when user-provided inputs fail validation."""


class DataFileNotFound(ExcogError):
    """Raised when a required data file or directory is not found."""


__all__ = [
    "ExcogError",
    "ConfigurationError",
    "InputValidationError",
    "DataFileNotFound",
]
