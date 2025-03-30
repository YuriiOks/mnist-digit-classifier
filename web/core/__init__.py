# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: core/__init__.py
# Description: Core package initialization
# Created: 2025-03-16
# Updated: 2025-03-24

"""Core functionality for the MNIST Digit Classifier."""

from core.app_state.session_state import SessionState
from core.app_state.theme_state import ThemeState
from core.app_state.navigation_state import NavigationState
from core.app_state import initialize_app_state
from core.errors import ErrorHandler, UIError, TemplateError, ComponentError
from core.database import DatabaseManager, db_manager, initialize_database

__all__ = [
    "SessionState",
    "ThemeState",
    "NavigationState",
    "initialize_app_state",
    "ErrorHandler",
    "UIError",
    "TemplateError",
    "ComponentError",
    "DatabaseManager",
    "db_manager",
    "initialize_database",
]
