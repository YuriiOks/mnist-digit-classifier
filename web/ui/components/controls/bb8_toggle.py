# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/controls/bb8_toggle.py
# Description: Improved BB8-themed toggle component for theme switching
# Created: 2025-03-17

import streamlit as st
from st_click_detector import click_detector
import logging
from typing import Optional, Callable, Dict, Any
import uuid

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager, theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

# Load BB8 CSS once at module level for better performance
BB8_INLINE_CSS = resource_manager.load_css(
    "components/controls/bb8-toggle.css"
)


class BB8Toggle(Component[Dict[str, Any]]):
    """BB8-themed toggle component for theme switching."""

    def __init__(
        self,
        theme_manager_instance: Optional[ThemeManager] = None,
        on_change: Optional[Callable[[str], None]] = None,
        *,
        key: str = "bb8_toggle",
        id: Optional[str] = None,
        classes: Optional[list] = None,
        attributes: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize the BB8Toggle component.

        Args:
            theme_manager_instance: ThemeManager instance to use
            on_change: Callback for theme changes
            key: Unique key for the component
            id: HTML ID for the component
            classes: CSS classes to apply
            attributes: HTML attributes to apply
            **kwargs: Additional arguments for the base Component
        """
        super().__init__(
            component_type="controls",
            component_name="bb8_toggle",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs,
        )

        self.__theme_manager = theme_manager_instance or theme_manager
        self.__on_change = on_change
        self.__key = key

        # Ensure BB8 CSS is loaded
        if not BB8_INLINE_CSS:
            self._logger.warning(
                "BB8 toggle CSS not found. The toggle may not display correctly."
            )

    def render(self):
        return super().render()

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> Dict[str, Any]:
        """
        Display the BB8 toggle with inline styles and handle theme changes.

        Returns:
            Dict containing theme state information
        """
        current_theme = self.__theme_manager.get_current_theme()
        is_dark = current_theme == "dark"

        # Unique ID to avoid conflicts when multiple toggles exist
        wrapper_id = f"bb8-toggle-{uuid.uuid4().hex[:8]}"

        # Define checked attribute based on current theme
        checked_attr = "checked" if is_dark else ""

        # If we have the BB8 CSS, use it. Otherwise, use a basic toggle.
        if BB8_INLINE_CSS:
            # Load the template
            template = resource_manager.load_template(
                "components/controls/bb8-toggle.html"
            )

            if template:
                # Inject the CSS and checked state
                full_html = template.replace(
                    "{BB8_INLINE_CSS}", BB8_INLINE_CSS
                )
                full_html = full_html.replace("{checked_attr}", checked_attr)

                # Use click detector to handle toggle clicks
                clicked = click_detector(
                    full_html, key=f"{self.__key}_{current_theme}"
                )

                if clicked == "bb8-toggle":
                    # Toggle theme
                    new_theme = "light" if is_dark else "dark"
                    self.__theme_manager.apply_theme(new_theme)

                    # Call change callback if provided
                    if self.__on_change:
                        self.__on_change(new_theme)

                    # Force rerun to update UI
                    st.rerun()
            else:
                self._logger.error("BB8 toggle template not found")
                # Fallback to basic toggle
                is_dark = st.toggle(
                    "Dark Mode", value=is_dark, key=f"{self.__key}_fallback"
                )
                if is_dark != (current_theme == "dark"):
                    new_theme = "dark" if is_dark else "light"
                    self.__theme_manager.apply_theme(new_theme)
                    if self.__on_change:
                        self.__on_change(new_theme)
                    st.rerun()
        else:
            # Fallback to basic toggle if CSS not available
            is_dark = st.toggle(
                "Dark Mode", value=is_dark, key=f"{self.__key}_fallback"
            )
            if is_dark != (current_theme == "dark"):
                new_theme = "dark" if is_dark else "light"
                self.__theme_manager.apply_theme(new_theme)
                if self.__on_change:
                    self.__on_change(new_theme)
                st.rerun()

        # Return theme state information
        return {"theme": current_theme, "is_dark": is_dark, "key": self.__key}
