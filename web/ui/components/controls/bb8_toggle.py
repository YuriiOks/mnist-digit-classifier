# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/bb8_toggle.py
# Description: BB8-themed toggle component for theme switching
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, Callable, Dict, Any

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager, theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

# Default BB8 toggle HTML as fallback
BB8_HTML = """
<label class="bb8-toggle">
    <input class="bb8-toggle__checkbox" type="checkbox">
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
"""


class BB8Toggle(Component[Dict[str, Any]]):
    """BB8-themed toggle component to switch between themes."""

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
        """
        Initialize the BB8Toggle component.

        Args:
            theme_manager_instance: Manages theme changes.
            on_change: Callback function when theme changes.
            key: Streamlit unique key.
            id: HTML ID for the component.
            classes: List of CSS classes to apply.
            attributes: Dictionary of HTML attributes.
            **kwargs: Additional keyword arguments.
        """
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
    def _load_bb8_toggle_css(self) -> None:
        """Load the BB8 toggle CSS."""
        css_content = resource_manager.load_css("components/controls/bb8-toggle.css")
        if css_content:
            resource_manager.inject_css(css_content)
            self._logger.debug("Loaded BB8 toggle CSS")
        else:
            self._logger.warning("Could not load BB8 toggle CSS")

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> str:
        """
        Render BB8 toggle component to HTML with embedded JS.
        
        Returns:
            HTML representation of the BB8 toggle.
        """
        # First try to load the template
        template_content = None
        for template_path in [
            "components/controls/bb8-toggle.html",
            "controls/bb8-toggle.html"
        ]:
            template_content = self.load_template(template_path)
            if template_content:
                self._logger.debug(f"Loaded BB8 template from: {template_path}")
                break
        
        # If no template was found, use the embedded fallback
        if not template_content:
            self._logger.debug("Using embedded BB8 HTML template")
            template_content = BB8_HTML
        
        # Set checked state based on current theme
        is_dark_mode = self.__theme_manager.is_dark_mode()
        html = template_content.replace(
            'type="checkbox"',
            f'type="checkbox" {"checked" if is_dark_mode else ""}'
        )

        # Add JavaScript for theme toggling
        js_code = self.__get_toggle_js()
        
        # Combine HTML and JS in a container
        combined_html = f"<div class='bb8-toggle-container'>{html}{js_code}</div>"
        
        return combined_html

    def __get_toggle_js(self) -> str:
        """
        JavaScript code to handle theme toggling.
        
        Returns:
            JavaScript as a string.
        """
        return """
        <script>
        (function() {
            const checkbox = document.querySelector('.bb8-toggle__checkbox');
            if (!checkbox) return;

            checkbox.addEventListener('change', function() {
                const theme = this.checked ? 'dark' : 'light';
                document.documentElement.setAttribute('data-theme', theme);
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: {
                        theme,
                        bb8_toggle_checkbox: this.checked
                    }
                }, '*');
            });
        })();
        </script>
        """

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> Dict[str, Any]:
        """
        Display BB8 toggle in Streamlit and handle theme changes.
        
        Returns:
            Dict containing theme status information.
        """
        # Load the CSS first
        self._load_bb8_toggle_css()
        
        # Render the component
        html = self.render()
        
        # Create a placeholder for returning values
        return_value = {}
        
        # Use components.v1.html for isolated rendering
        st.components.v1.html(html, height=120, key=f"{self.__key}_html")
        
        # Process theme change if present in session state
        if f"{self.__key}_html" in st.session_state:
            component_value = st.session_state[f"{self.__key}_html"]
            if isinstance(component_value, dict) and "theme" in component_value:
                # Get the new theme
                new_theme = component_value["theme"]
                is_checked = component_value.get("bb8_toggle_checkbox", new_theme == "dark")
                
                # Update return value
                return_value = {
                    "theme": new_theme,
                    "is_dark": new_theme == "dark",
                    "changed": True
                }
                
                # Apply theme change if different
                current_theme = self.__theme_manager.get_current_theme()
                if new_theme != current_theme:
                    self._logger.debug(f"Applying theme: {new_theme}")
                    self.__theme_manager.apply_theme(new_theme)
                    
                    # Call on_change callback if provided
                    if self.__on_change:
                        self.__on_change(new_theme)
            
        # If we have no theme change data yet, return current state
        if not return_value:
            current_theme = self.__theme_manager.get_current_theme()
            return_value = {
                "theme": current_theme,
                "is_dark": current_theme == "dark",
                "changed": False
            }
        
        return return_value