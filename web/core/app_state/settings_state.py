# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/settings_state.py
# Description: State management for application settings
# Created: 2024-05-01

import logging
from typing import Dict, Any, Optional

from core.app_state.session_state import SessionState
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class SettingsState:
    """Manage application settings state."""

    SETTINGS_KEY = "app_settings"

    _logger = logging.getLogger(f"{__name__}.SettingsState")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize default settings if not already present."""
        if not SessionState.has_key(cls.SETTINGS_KEY):
            default_settings = {
                # Theme settings
                "theme": {
                    "mode": "light",              # light or dark
                    "accent_color": "#6366F1",  # primary accent color
                    "font_family": "Poppins",     # primary font family
                    "enable_animations": True,    # enable UI animations
                },
                # Canvas settings
                "canvas": {
                    "canvas_size": 280,               # canvas size in pixels
                    "stroke_width": 15,               # line width
                    "stroke_color": "#000000",      # line color
                    "background_color": "#ffffff",  # canvas background
                    "enable_grid": False,             # show grid on canvas
                },
                # Prediction settings
                "prediction": {
                    "auto_predict": True,       # auto-predict on drawings
                    "show_confidence": True,    # show confidence percentages
                    "min_confidence": 0.5,      # minimum confidence threshold
                    "show_alternatives": True,  # show alternative predictions
                },
                # Application settings
                "app": {
                    "save_history": True,     # save prediction history
                    "max_history": 50,        # max number of history items
                    "show_tooltips": True,    # show tooltips on UI elements
                    "debug_mode": False,      # enable debug features
                },
            }
            SessionState.set(cls.SETTINGS_KEY, default_settings)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all application settings.

        Returns:
            Dict containing all application settings
        """
        cls.initialize()
        return SessionState.get(cls.SETTINGS_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_category(cls, category: str) -> Dict[str, Any]:
        """Get all settings for a specific category.

        Args:
            category: Setting category (theme, canvas, etc.)

        Returns:
            Dict containing category settings
        """
        cls.initialize()
        settings = SessionState.get(cls.SETTINGS_KEY)
        return settings.get(category, {})

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_setting(
        cls,
        category: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a specific setting value.

        Args:
            category: Setting category (theme, canvas, etc.)
            key: Setting key name
            default: Default value if setting doesn't exist

        Returns:
            Setting value or default
        """
        cls.initialize()
        settings = SessionState.get(cls.SETTINGS_KEY)

        if category not in settings or key not in settings[category]:
            return default

        return settings[category][key]

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_setting(cls, category: str, key: str, value: Any) -> None:
        """Set a specific setting value.

        Args:
            category: Setting category (theme, canvas, etc.)
            key: Setting key name
            value: New value to set
        """
        cls.initialize()
        settings = SessionState.get(cls.SETTINGS_KEY)
        if category not in settings:
            settings[category] = {}

        settings[category][key] = value
        SessionState.set(cls.SETTINGS_KEY, settings)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def update_category(cls, category: str, values: Dict[str, Any]) -> None:
        """Update an entire settings category.

        Args:
            category: Setting category (theme, canvas, etc.)
            values: Dictionary of all key-value pairs to update
        """
        cls.initialize()
        settings = SessionState.get(cls.SETTINGS_KEY)
        if category not in settings:
            settings[category] = {}

        settings[category].update(values)
        SessionState.set(cls.SETTINGS_KEY, settings)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def reset_to_defaults(cls, category: Optional[str] = None) -> None:
        """Reset settings to default values.

        Args:
            category: Optional category to reset (or all if None)
        """
        if category:
            # Just delete the specific category to trigger re-initialization
            settings = SessionState.get(cls.SETTINGS_KEY, {})
            if category in settings:
                del settings[category]
                SessionState.set(cls.SETTINGS_KEY, settings)
        else:
            # Reset all settings
            SessionState.set(cls.SETTINGS_KEY, None)

        # Re-initialize to ensure defaults are set
        cls.initialize()
