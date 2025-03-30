# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/registry/__init__.py
# Description: Core package initialization
# Created: 2025-03-16
# Updated: 2025-03-30

"""Core functionality for the MNIST Digit Classifier."""

from core.app_state import (
    SessionState,
    ThemeState,
    NavigationState,
    initialize_app_state,
)
from core.errors import ErrorHandler, UIError, TemplateError, ComponentError

__all__ = [
    "SessionState",
    "ThemeState",
    "NavigationState",
    "initialize_app_state",
    "ErrorHandler",
    "UIError",
    "TemplateError",
    "ComponentError",
]
