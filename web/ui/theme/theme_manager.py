# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/theme/theme_manager.py
# Description: Theme management
# Created: 2025-03-16
# Updated: 2025-03-30

import streamlit as st
import logging
from typing import Dict, Any, Optional, Literal

from core.app_state.theme_state import ThemeState
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)

# Type alias for theme options
ThemeMode = Literal["light", "dark"]


class ThemeManager:
    """
    Streamlined theme management for light/dark mode.

    This class handles theme switching, CSS variable injection, and theme-aware
    UI element styling.
    """

    # Theme constants
    LIGHT_THEME = "light"
    DARK_THEME = "dark"
    DEFAULT_THEME = LIGHT_THEME

    # State key constants
    CURRENT_THEME_KEY = "current_theme"
    THEME_DATA_KEY = "theme_data"

    def __init__(self):
        """Initialize the theme manager."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._themes: Dict[str, Dict[str, Any]] = {}
        self._load_themes()

    def _load_themes(self) -> None:
        """Load theme configurations from JSON files."""
        # Try to load light theme
        light_theme = resource_manager.load_json_resource("themes/light.json")
        if light_theme:
            self._themes[self.LIGHT_THEME] = light_theme
            self._logger.debug("Loaded light theme configuration")
        else:
            # Create default light theme
            self._themes[self.LIGHT_THEME] = self._get_default_light_theme()
            self._logger.warning("Using default light theme (config file not found)")

        # Try to load dark theme
        dark_theme = resource_manager.load_json_resource("themes/dark.json")
        if dark_theme:
            self._themes[self.DARK_THEME] = dark_theme
            self._logger.debug("Loaded dark theme configuration")
        else:
            # Create default dark theme
            self._themes[self.DARK_THEME] = self._get_default_dark_theme()
            self._logger.warning("Using default dark theme (config file not found)")

    def _get_default_light_theme(self) -> Dict[str, Any]:
        """
        Get default light theme configuration.

        Returns:
            Default light theme configuration dictionary.
        """
        return {
            "name": "light",
            "displayName": "Light",
            "colors": {
                "primary": "#4361ee",
                "primary-rgb": "67, 97, 238",
                "secondary": "#4cc9f0",
                "secondary-rgb": "76, 201, 240",
                "accent": "#4895ef",
                "accent-rgb": "72, 149, 239",
                "background": "#f8f9fa",
                "background-alt": "#f1f3f5",
                "card": "#ffffff",
                "card-alt": "#f1f3f5",
                "text": "#212529",
                "text-light": "#6c757d",
                "text-muted": "#adb5bd",
                "border": "#dee2e6",
                "success": "#4cc9f0",
                "warning": "#fbbf24",
                "error": "#f87171",
                "info": "#60a5fa",
            },
            "fonts": {
                "primary": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "heading": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "code": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            },
            "settings": {
                "border-radius": "0.5rem",
                "shadow-strength": "1.0",
                "animations-enabled": True,
            },
        }

    def _get_default_dark_theme(self) -> Dict[str, Any]:
        """
        Get default dark theme configuration.

        Returns:
            Default dark theme configuration dictionary.
        """
        return {
            "name": "dark",
            "displayName": "Dark",
            "colors": {
                "primary": "#ee4347",
                "primary-rgb": "238, 67, 71",
                "secondary": "#f0c84c",
                "secondary-rgb": "240, 200, 76",
                "accent": "#5e81f4",
                "accent-rgb": "94, 129, 244",
                "background": "#121212",
                "background-alt": "#1a1a1a",
                "card": "#1e1e1e",
                "card-alt": "#252525",
                "text": "#f8f9fa",
                "text-light": "#d1d5db",
                "text-muted": "#9ca3af",
                "border": "#383838",
                "success": "#34d399",
                "warning": "#fbbf24",
                "error": "#f87171",
                "info": "#60a5fa",
            },
            "fonts": {
                "primary": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "heading": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                "code": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            },
            "settings": {
                "border-radius": "0.5rem",
                "shadow-strength": "1.4",
                "animations-enabled": True,
            },
        }

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(self) -> None:
        """Initialize theme state."""
        # Set default theme if not already set
        if self.CURRENT_THEME_KEY not in st.session_state:
            st.session_state[self.CURRENT_THEME_KEY] = self.DEFAULT_THEME
            self._logger.debug(f"Set default theme: {self.DEFAULT_THEME}")

        # Apply the current theme
        self.apply_theme(st.session_state[self.CURRENT_THEME_KEY])

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_current_theme(self) -> str:
        """
        Get the current theme mode.

        Returns:
            Current theme mode ("light" or "dark").
        """
        if self.CURRENT_THEME_KEY not in st.session_state:
            st.session_state[self.CURRENT_THEME_KEY] = self.DEFAULT_THEME

        return st.session_state[self.CURRENT_THEME_KEY]

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def is_dark_mode(self) -> bool:
        """
        Check if dark mode is active.

        Returns:
            True if dark mode is active, False otherwise.
        """
        return self.get_current_theme() == self.DARK_THEME

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_theme_data(self, theme_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Get theme data for the specified mode.

        Args:
            theme_mode: Theme mode to get data for. Defaults to current theme.

        Returns:
            Theme data dictionary.
        """
        theme_mode = theme_mode or self.get_current_theme()

        # Make sure we're using a valid theme mode
        if theme_mode not in [self.LIGHT_THEME, self.DARK_THEME]:
            theme_mode = self.DEFAULT_THEME

        return self._themes.get(theme_mode, self._themes[self.DEFAULT_THEME])

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def toggle_theme(self) -> str:
        """
        Toggle between light and dark themes.

        Returns:
            New theme mode after toggling.
        """
        current_theme = self.get_current_theme()
        new_theme = (
            self.DARK_THEME if current_theme == self.LIGHT_THEME else self.LIGHT_THEME
        )

        # Apply the new theme
        self.apply_theme(new_theme)

        # Log the change
        self._logger.info(f"Theme toggled from {current_theme} to {new_theme}")

        return new_theme

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def apply_theme(self, theme_mode: str) -> None:
        """
        Apply the specified theme.

        Args:
            theme_mode: Theme mode to apply ("light" or "dark").
        """
        # Make sure we're using a valid theme mode
        if theme_mode not in [self.LIGHT_THEME, self.DARK_THEME]:
            theme_mode = self.DEFAULT_THEME

        # Log theme change
        self._logger.debug(f"Applying theme: {theme_mode}")

        # Set the current theme in session state
        st.session_state[self.CURRENT_THEME_KEY] = theme_mode

        # Set theme data in session state for easy access by components
        theme_data = self.get_theme_data(theme_mode)
        st.session_state[self.THEME_DATA_KEY] = theme_data

        # Apply CSS variables
        self._apply_css_variables(theme_data)

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def _apply_css_variables(self, theme_data: Dict[str, Any]) -> None:
        """
        Apply CSS variables from theme data.

        Args:
            theme_data: Theme data dictionary.
        """
        # Generate CSS variables from theme data
        css_vars = [":root {"]

        # Add color variables
        colors = theme_data.get("colors", {})
        for name, value in colors.items():
            css_vars.append(f"  --color-{name}: {value};")

        # Add font variables
        fonts = theme_data.get("fonts", {})
        for name, value in fonts.items():
            css_vars.append(f"  --font-{name}: {value};")

        # Add settings variables
        settings = theme_data.get("settings", {})
        for name, value in settings.items():
            # Convert boolean values to strings
            if isinstance(value, bool):
                value = str(value).lower()
            css_vars.append(f"  --{name}: {value};")

        css_vars.append("}")

        # Inject CSS variables
        css = "\n".join(css_vars)
        print(f"{css[:20]} to be continued...")
        resource_manager.inject_css(css)

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_theme_color(self, color_name: str, for_inline: bool = False) -> str:
        """
        Get a color from the current theme.

        Args:
            color_name: Name of the color to get.
            for_inline: Whether to return CSS var format for inline styles.

        Returns:
            The color value or CSS variable reference.
        """
        theme_data = self.get_theme_data()
        colors = theme_data.get("colors", {})

        # For inline styles, return the actual color value if available
        if for_inline and color_name in colors:
            return colors[color_name]

        # Otherwise return a CSS variable reference
        return f"var(--color-{color_name})"

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_all_colors(self) -> Dict[str, str]:
        """
        Get all colors from the current theme.

        Returns:
            Dictionary of color names and values.
        """
        theme_data = self.get_theme_data()
        return theme_data.get("colors", {})

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def detect_system_preference(self) -> None:
        """Detect and apply system color scheme preference."""
        # Use JavaScript to detect system preference
        js_code = """
        <script>
            (function() {
                // Check if prefers-color-scheme media query is supported
                if (window.matchMedia) {
                    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    const theme = prefersDark ? 'dark' : 'light';
                    
                    // Only apply if user hasn't set a preference
                    if (!localStorage.getItem('theme-preference')) {
                        document.documentElement.setAttribute('data-theme', theme);
                        
                        // Pass preference to Streamlit
                        const themeData = {
                            theme: theme,
                            system_detected: true
                        };
                        window.parent.postMessage({
                            type: "streamlit:setComponentValue",
                            value: themeData
                        }, "*");
                    }
                }
            })();
        </script>
        """
        st.components.v1.html(js_code, height=0)


# Create a singleton instance
theme_manager = ThemeManager()
