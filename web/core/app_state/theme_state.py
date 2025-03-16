# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/theme_state.py
# Description: Theme-specific state management
# Created: 2025-03-16

import logging
from typing import Dict, Any, Literal, Optional

from core.app_state.session_state import SessionState
from utils.aspects import AspectUtils
from utils.resource_manager import resource_manager

logger = logging.getLogger(__name__)


class ThemeState:
    """
    Theme-specific state management.

    This class provides themed state management for the application,
    handling theme switching, color schemes, and related settings.
    """

    # Theme state key prefixes
    PREFIX = "theme_"
    CURRENT_THEME_KEY = f"{PREFIX}current"
    THEME_COLORS_KEY = f"{PREFIX}colors"
    THEME_FONTS_KEY = f"{PREFIX}fonts"
    THEME_SETTINGS_KEY = f"{PREFIX}settings"

    # Available themes
    THEME_LIGHT = "light"
    THEME_DARK = "dark"

    # Default theme settings
    DEFAULT_THEME = THEME_LIGHT
    DEFAULT_COLORS = {
        THEME_LIGHT: {
            "primary": "#4361ee",
            "secondary": "#4cc9f0",
            "background": "#FFFFFF",
            "surface": "#F5F5F5",
            "text": "#333333",
            "text_secondary": "#555555",
            "border": "#DDDDDD",
            "error": "#FF3B30",
            "success": "#34C759",
            "warning": "#FFCC00"
        },
        THEME_DARK: {
            "primary": "#7B7BFF",
            "secondary": "#40C7D7",
            "background": "#121212",
            "surface": "#1E1E1E",
            "text": "#FFFFFF",
            "text_secondary": "#BBBBBB",
            "border": "#333333",
            "error": "#FF453A",
            "success": "#32D74B",
            "warning": "#FFD60A"
        }
    }
    DEFAULT_FONTS = {
        "heading": "system-ui, sans-serif",
        "body": "system-ui, sans-serif",
        "code": "monospace"
    }
    DEFAULT_SETTINGS = {
        "animations_enabled": True,
        "transitions_speed": "normal",
        "border_radius": "4px"
    }

    _logger = logging.getLogger(f"{__name__}.ThemeState")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def _load_theme_config(cls) -> Dict[str, Any]:
        """
        Load theme configurations from JSON files.

        Returns:
            Dict containing the theme configurations.
        """
        theme_config = {
            "colors": {},
            "fonts": cls.DEFAULT_FONTS.copy(),
            "settings": cls.DEFAULT_SETTINGS.copy()
        }

        try:
            # Load default config (fonts and settings)
            default_config = resource_manager.load_json_resource("config/themes/default.json")

            if default_config:
                if "fonts" in default_config:
                    theme_config["fonts"].update(default_config["fonts"])
                if "settings" in default_config:
                    theme_config["settings"].update(default_config["settings"])

            # Load light theme colors
            light_config = resource_manager.load_json_resource("config/themes/light.json")

            if light_config and "colors" in light_config:
                theme_config["colors"][cls.THEME_LIGHT] = light_config["colors"]

            # Load dark theme colors
            dark_config = resource_manager.load_json_resource("config/themes/dark.json")

            if dark_config and "colors" in dark_config:
                theme_config["colors"][cls.THEME_DARK] = dark_config["colors"]
        except Exception as e:
            cls._logger.error(f"Error loading theme config: {e}")
            
        # Ensure we have fallback values for colors
        if cls.THEME_LIGHT not in theme_config["colors"]:
            theme_config["colors"][cls.THEME_LIGHT] = cls.DEFAULT_COLORS[cls.THEME_LIGHT]
        if cls.THEME_DARK not in theme_config["colors"]:
            theme_config["colors"][cls.THEME_DARK] = cls.DEFAULT_COLORS[cls.THEME_DARK]

        return theme_config

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize the theme state."""
        try:
            # Always load the theme configuration
            theme_config = cls._load_theme_config()

            # Set the current theme (if not already set)
            if not SessionState.has_key(cls.CURRENT_THEME_KEY):
                SessionState.set(cls.CURRENT_THEME_KEY, cls.DEFAULT_THEME)

            # Set theme colors using the loaded config
            if not SessionState.has_key(cls.THEME_COLORS_KEY):
                colors = theme_config.get("colors", cls.DEFAULT_COLORS)
                SessionState.set(cls.THEME_COLORS_KEY, colors)

            # Set theme fonts using the loaded config
            if not SessionState.has_key(cls.THEME_FONTS_KEY):
                fonts = theme_config.get("fonts", cls.DEFAULT_FONTS)
                SessionState.set(cls.THEME_FONTS_KEY, fonts)

            # Set theme settings using the loaded config
            if not SessionState.has_key(cls.THEME_SETTINGS_KEY):
                settings = theme_config.get("settings", cls.DEFAULT_SETTINGS)
                SessionState.set(cls.THEME_SETTINGS_KEY, settings)
        except Exception as e:
            cls._logger.error(f"Error initializing theme state: {e}")
            # Fallback defaults
            if not SessionState.has_key(cls.CURRENT_THEME_KEY):
                SessionState.set(cls.CURRENT_THEME_KEY, cls.DEFAULT_THEME)
            if not SessionState.has_key(cls.THEME_COLORS_KEY):
                SessionState.set(cls.THEME_COLORS_KEY, cls.DEFAULT_COLORS)
            if not SessionState.has_key(cls.THEME_FONTS_KEY):
                SessionState.set(cls.THEME_FONTS_KEY, cls.DEFAULT_FONTS)
            if not SessionState.has_key(cls.THEME_SETTINGS_KEY):
                SessionState.set(cls.THEME_SETTINGS_KEY, cls.DEFAULT_SETTINGS)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_current_theme(cls) -> str:
        """
        Get the current theme.

        Returns:
            str: The current theme (e.g., "light", "dark").
        """
        cls.initialize()
        return SessionState.get(cls.CURRENT_THEME_KEY, cls.DEFAULT_THEME)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_theme(cls, theme: str) -> None:
        """
        Set the current theme.

        Args:
            theme: The theme to set (e.g., "light", "dark").
        """
        cls.initialize()

        # Validate theme
        if theme not in [cls.THEME_LIGHT, cls.THEME_DARK]:
            theme = cls.DEFAULT_THEME

        SessionState.set(cls.CURRENT_THEME_KEY, theme)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def toggle_theme(cls) -> str:
        """
        Toggle between light and dark themes.

        Returns:
            str: The new theme after toggling.
        """
        current = cls.get_current_theme()
        new_theme = cls.THEME_DARK if current == cls.THEME_LIGHT else cls.THEME_LIGHT
        cls.set_theme(new_theme)
        return new_theme

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def is_dark_mode(cls) -> bool:
        """
        Check if dark mode is active.

        Returns:
            bool: True if dark mode is active, False otherwise.
        """
        return cls.get_current_theme() == cls.THEME_DARK

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_color(cls, color_name: str) -> str:
        """
        Get a color for the current theme.

        Args:
            color_name: Name of the color to retrieve.

        Returns:
            str: The color value (e.g., "#ffffff").
        """
        theme = cls.get_current_theme()
        colors = SessionState.get(cls.THEME_COLORS_KEY, cls.DEFAULT_COLORS)

        theme_colors = colors.get(theme, {})
        return theme_colors.get(color_name, "#000000")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_all_colors(cls) -> Dict[str, str]:
        """
        Get all colors for the current theme.

        Returns:
            Dict[str, str]: Dictionary of color names and values.
        """
        theme = cls.get_current_theme()
        colors = SessionState.get(cls.THEME_COLORS_KEY, cls.DEFAULT_COLORS)
        return colors.get(theme, {})

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_css_variables(cls) -> str:
        """
        Get CSS variables for the current theme.

        Returns:
            str: CSS variable definitions.
        """
        colors = cls.get_all_colors()
        fonts = SessionState.get(cls.THEME_FONTS_KEY, cls.DEFAULT_FONTS)
        settings = SessionState.get(cls.THEME_SETTINGS_KEY, cls.DEFAULT_SETTINGS)

        css_vars = [":root {"]

        # Add color variables
        for name, value in colors.items():
            css_vars.append(f"  --color-{name}: {value};")

        # Add font variables
        for name, value in fonts.items():
            css_vars.append(f"  --font-{name}: {value};")

        # Add settings variables
        for name, value in settings.items():
            # Convert boolean values to strings
            if isinstance(value, bool):
                value = str(value).lower()
            css_vars.append(f"  --{name.replace('_', '-')}: {value};")

        css_vars.append("}")
        return "\n".join(css_vars)