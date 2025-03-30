# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/app_state/navigation_state.py
# Description: Navigation state management
# Created: 2025-03-16
# Updated: 2025-03-30

import logging
from typing import Dict, List, Optional, Any

from core.app_state.session_state import SessionState
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class NavigationState:
    """
    Manage application navigation state.

    This class provides state management for navigation, tracking the current view
    and navigation history.
    """

    # Session state keys
    ACTIVE_VIEW_KEY = "active_view"
    NAV_HISTORY_KEY = "navigation_history"
    ROUTES_KEY = "navigation_routes"

    # Default view
    DEFAULT_VIEW = "home"

    # Default routes
    DEFAULT_ROUTES = [
        {"id": "home", "label": "Home", "icon": "🏠"},
        {"id": "draw", "label": "Draw", "icon": "✏️"},
        {"id": "history", "label": "History", "icon": "📊"},
        {"id": "settings", "label": "Settings", "icon": "⚙️"},
    ]

    _logger = logging.getLogger(f"{__name__}.NavigationState")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize navigation state with defaults if not already present."""
        # Set default active view if not set
        if not SessionState.has_key(cls.ACTIVE_VIEW_KEY):
            SessionState.set(cls.ACTIVE_VIEW_KEY, cls.DEFAULT_VIEW)

        # Initialize navigation history if not set
        if not SessionState.has_key(cls.NAV_HISTORY_KEY):
            SessionState.set(cls.NAV_HISTORY_KEY, [cls.DEFAULT_VIEW])

        # Initialize routes if not set
        if not SessionState.has_key(cls.ROUTES_KEY):
            SessionState.set(cls.ROUTES_KEY, cls.DEFAULT_ROUTES)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_active_view(cls) -> str:
        """
        Get the currently active view.

        Returns:
            str: ID of the active view
        """
        cls.initialize()
        active_view = SessionState.get(cls.ACTIVE_VIEW_KEY, cls.DEFAULT_VIEW)
        return active_view

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_active_view(cls, view_id: str) -> None:
        """
        Set the active view and update navigation history.

        Args:
            view_id: ID of the view to set as active
        """
        cls.initialize()

        # Update active view
        SessionState.set(cls.ACTIVE_VIEW_KEY, view_id)

        # Update navigation history
        history = SessionState.get(cls.NAV_HISTORY_KEY, [])

        # Don't add duplicate consecutive entries
        if not history or history[-1] != view_id:
            history.append(view_id)
            # Limit history size to prevent memory issues
            if len(history) > 20:
                history = history[-20:]
            SessionState.set(cls.NAV_HISTORY_KEY, history)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_nav_history(cls) -> List[str]:
        """
        Get the navigation history.

        Returns:
            List[str]: List of view IDs in navigation history
        """
        cls.initialize()
        history = SessionState.get(cls.NAV_HISTORY_KEY, [])
        return history

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_routes(cls) -> List[Dict[str, Any]]:
        """
        Get all available navigation routes.

        Returns:
            List[Dict[str, Any]]: List of route definitions
        """
        cls.initialize()
        routes = SessionState.get(cls.ROUTES_KEY, cls.DEFAULT_ROUTES)
        return routes

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def go_back(cls) -> str:
        """
        Navigate back to the previous view.

        Returns:
            str: ID of the previous view, or current view if history is empty
        """
        cls.initialize()
        history = SessionState.get(cls.NAV_HISTORY_KEY, [])

        # Can't go back if history is empty or only has one item
        if len(history) <= 1:
            return cls.get_active_view()

        # Remove current view
        history.pop()

        # Set previous view as current
        previous = history[-1]
        SessionState.set(cls.ACTIVE_VIEW_KEY, previous)
        SessionState.set(cls.NAV_HISTORY_KEY, history)

        return previous
