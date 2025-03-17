# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/bb8_toggle.py
# Description: BB8-themed toggle component for theme switching
# Created: 2025-03-16

import streamlit as st
from st_click_detector import click_detector
import logging
from typing import Optional, Callable, Dict, Any
import uuid

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager, theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils
from typing import Optional, Callable, Dict, Any

BB8_INLINE_CSS = resource_manager.load_css("assets/css/components/controls/bb8-toggle.css")

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
        **kwargs
    ):
        """Initialize the BB8Toggle component."""
        super().__init__(
            component_type="controls",
            component_name="bb8_toggle",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )

        self.__theme_manager = theme_manager_instance or theme_manager
        self.__on_change = on_change
        self.__key = key

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> str:
        """Render the BB8 toggle component."""
        # Get current theme
        current_theme = self.__theme_manager.get_current_theme()
        is_dark = self.__theme_manager.is_dark_mode()
        
        # Simple inline styled version (we're not using the template anymore)
        return f""

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> Dict[str, Any]:
        """
        Display the BB8 toggle with inline styles and st_click_detector.
        This approach ensures all CSS is inlined with the returned HTML.
        """
        current_theme = self.__theme_manager.get_current_theme()
        is_dark = (current_theme == "dark")

        # Unique ID so multiple toggles don't conflict
        wrapper_id = f"bb8-toggle-{uuid.uuid4().hex[:8]}"

        checked_attr = "checked" if is_dark else ""

        # 2) Inline all the CSS along with the BB8 HTML:
        full_html = f"""
        <style>
        {BB8_INLINE_CSS}
        </style>
        <div id="{wrapper_id}" style="text-align:center; padding-top:20px; padding-bottom:20px;">
          <a href="#" id="bb8-toggle" style="display:inline-block; text-decoration:none;">
            <label class="bb8-toggle">
              <input class="bb8-toggle__checkbox" type="checkbox" {checked_attr} />
              <div class="bb8-toggle__container">
                <div class="bb8-toggle__scenery">
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="bb8-toggle__star"></div>
                  <div class="tatto-1"></div>
                  <div class="tatto-2"></div>
                  <div class="gomrassen"></div>
                  <div class="hermes"></div>
                  <div class="chenini"></div>
                  <div class="bb8-toggle__cloud"></div>
                  <div class="bb8-toggle__cloud"></div>
                  <div class="bb8-toggle__cloud"></div>
                </div>
                <div class="bb8">
                  <div class="bb8__head-container">
                    <div class="bb8__antenna"></div>
                    <div class="bb8__antenna"></div>
                    <div class="bb8__head"></div>
                  </div>
                  <div class="bb8__body"></div>
                </div>
                <div class="artificial__hidden">
                  <div class="bb8__shadow"></div>
                </div>
              </div>
            </label>
          </a>
        </div>
        """

        # Use st_click_detector to see if user clicked on the anchor with id="bb8-toggle"
        clicked = click_detector(full_html, key=f"{self.__key}_{current_theme}")

        if clicked == "bb8-toggle":
            new_theme = "light" if is_dark else "dark"
            self.__theme_manager.apply_theme(new_theme)

            if self.__on_change:
                self.__on_change(new_theme)

            st.rerun()

        return {
            "theme": current_theme,
            "is_dark": is_dark,
            "key": self.__key
        }