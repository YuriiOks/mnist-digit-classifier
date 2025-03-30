# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/app_state/__init__.py
# Description: App state package initialization
# Created: 2025-03-16
# Updated: 2025-03-30

"""Application state management for the MNIST Digit Classifier."""

from core.app_state.session_state import SessionState
from core.app_state.theme_state import ThemeState
from core.app_state.navigation_state import NavigationState

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SessionState",
    "ThemeState",
    "NavigationState",
    "initialize_app_state",
]


def initialize_app_state():
    """Initialize the application state."""
    logger.info("Initializing app state")

    # Initialize all state components
    SessionState.initialize()

    try:
        ThemeState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize ThemeState: {e}")

    try:
        NavigationState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize NavigationState: {e}")
