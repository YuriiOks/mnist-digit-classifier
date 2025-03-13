# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/navigation_state.py
# Description: Navigation state management
# Created: 2024-05-01

import logging
from typing import Dict, Any, Optional, List

from core.app_state.session_state import SessionState
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class NavigationState:
    """Manage application navigation state."""

    # Session state keys - RENAMED to avoid widget conflicts
    ACTIVE_VIEW_KEY = "active_view"
    NAV_HISTORY_KEY = "navigation_history"  # Changed from "nav_history"
    ROUTES_KEY = "navigation_routes"        # Changed from "routes"

    # Default view
    DEFAULT_VIEW = "home"

    # Default routes - SIMPLIFIED to exactly 4 tabs
    DEFAULT_ROUTES = [
        {"id": "home", "label": "Home", "icon": "🏠"},
        {"id": "draw", "label": "Draw", "icon": "✏️"},
        {"id": "history", "label": "History", "icon": "📊"},
        {"id": "settings", "label": "Settings", "icon": "⚙️"}
    ]

    _logger = logging.getLogger(f"{__name__}.NavigationState")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:

        cls._logger = logging.getLogger(f"{__name__}.{cls.__class__.__name__}")
        cls._logger.debug(f"Initializing {cls.__class__.__name__} instance")

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
        else:
            # Force update to the simplified routes - using set()
            # instead of direct assignment
            SessionState.set(cls.ROUTES_KEY, cls.DEFAULT_ROUTES)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_active_view(cls) -> str:
        """Get the currently active view.

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
        """Set the active view and update navigation history.

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
        """Get the navigation history.

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
        """Get all available navigation routes.

        Returns:
            List[Dict[str, Any]]: List of route definitions
        """
        cls.initialize()
        routes = SessionState.get(cls.ROUTES_KEY, cls.DEFAULT_ROUTES)
        return routes

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def add_route(cls, route: Dict[str, Any]) -> None:
        """Add a new navigation route.

        Args:
            route: Route definition dictionary with id, label, and icon
        """
        cls.initialize()
        routes = SessionState.get(cls.ROUTES_KEY, cls.DEFAULT_ROUTES)

        # Check if route already exists
        for i, existing_route in enumerate(routes):
            if existing_route.get("id") == route.get("id"):
                # Update existing route
                routes[i] = route
                SessionState.set(cls.ROUTES_KEY, routes)
                return

        # Add new route
        routes.append(route)
        SessionState.set(cls.ROUTES_KEY, routes)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def go_back(cls) -> str:
        """Navigate back to the previous view.

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
