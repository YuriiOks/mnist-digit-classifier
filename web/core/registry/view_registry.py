# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/registry/view_registry.py
# Description: Registry for application views
# Created: 2024-05-01

import logging
from typing import Dict, List, Type, Optional
from utils.aspects import AspectUtils
from ui.views.base_view import BaseView

logger = logging.getLogger(__name__)


class ViewRegistry:
    """Registry and factory for application views.

    This registry manages the association between view IDs and their
    corresponding view classes. It allows lazy instantiation of views.
    """

    # Class variable storing registered view classes
    _view_classes: Dict[str, Type[BaseView]] = {}
    # Add a class logger
    _logger = logging.getLogger(__name__)

    @classmethod
    @AspectUtils.log_method
    def register_view(cls, view_class: Type[BaseView]) -> None:
        """Register a view class.

        Args:
            view_class: The view class to register.
        """
        if not hasattr(view_class, 'view_id'):
            # Use the class logger instead of instance logger
            cls._logger.error(
                f"Cannot register {view_class.__name__}: missing 'view_id'."
            )
            return

        view_id = view_class().view_id
        cls._logger.debug(f"Registering view class '{view_class.__name__}' as '{view_id}'")
        cls._view_classes[view_id] = view_class

    @classmethod
    @AspectUtils.log_method
    def get_view(cls, view_id: str) -> Optional[BaseView]:
        """Instantiate and retrieve a view by its ID.

        Args:
            view_id: The identifier of the view to instantiate.

        Returns:
            An instance of the requested view, or None if not found.
        """
        view_class = cls._view_classes.get(view_id)
        if view_class:
            cls._logger.debug(f"Instantiating view '{view_id}'")
            return view_class()
        else:
            cls._logger.warning(f"Requested view '{view_id}' not registered.")
            return None

    @classmethod
    def get_registered_view_ids(cls) -> List[str]:
        """Retrieve all registered view IDs.

        Returns:
            List of registered view IDs.
        """
        return list(cls._view_classes.keys())

    @classmethod
    @AspectUtils.log_method
    def register_default_views(cls) -> None:
        """Register all default application views."""
        cls._logger.debug("Registering default views")

        # Import views here to prevent circular dependencies
        from ui.views.home.home_view import HomeView
        from ui.views.draw.draw_view import DrawView
        from ui.views.history.history_view import HistoryView
        from ui.views.settings.settings_view import SettingsView

        default_views = [HomeView, DrawView, HistoryView, SettingsView]

        for view_cls in default_views:
            cls.register_view(view_cls)

        # Optionally register debug view (if available)
        try:
            from ui.views.debug.debug_view import DebugView
            cls.register_view(DebugView)
            cls._logger.debug("Debug view registered successfully.")
        except ImportError:
            cls._logger.debug("Debug view unavailable; skipping.")