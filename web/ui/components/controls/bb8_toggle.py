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
        # Try multiple potential paths for the CSS
        potential_paths = [
            "components/controls/bb8-toggle.css",
            "components/controls/bb8_toggle.css",
            "controls/bb8-toggle.css",
            "controls/bb8_toggle.css"
        ]
        
        css_loaded = False
        for css_path in potential_paths:
            css_content = resource_manager.load_css(css_path)
            if css_content:
                resource_manager.inject_css(css_content)
                self._logger.debug(f"Successfully loaded BB8 toggle CSS from {css_path}")
                css_loaded = True
                break
        
        if not css_loaded:
            # If no CSS was loaded, inject the minimum CSS needed
            self._logger.warning("Could not load BB8 toggle CSS from files, using inline minimum CSS")
            min_css = """
            .bb8-toggle {
                display: inline-block;
                cursor: pointer;
            }
            .bb8-toggle__container {
                width: 170px;
                height: 90px;
                background: linear-gradient(#2c4770, #070e2b 35%, #628cac 50% 70%, #a6c5d4);
                border-radius: 99em;
                position: relative;
                transition: 0.4s;
            }
            .bb8-toggle__checkbox {
                display: none;
            }
            .bb8 {
                position: absolute;
                left: 15px;
                top: 15px;
                transition: 0.4s;
            }
            .bb8__body {
                width: 60px;
                height: 60px;
                background: white;
                border-radius: 50%;
            }
            .bb8__head {
                width: 40px;
                height: 25px;
                background: white;
                border-radius: 25px 25px 0 0;
                margin-bottom: -3px;
            }
            .bb8-toggle__checkbox:checked + .bb8-toggle__container .bb8 {
                left: calc(100% - 75px);
            }
            """
            resource_manager.inject_css(min_css)

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> str:
        """Render the BB8 toggle component.
        
        Returns:
            HTML representation of the BB8 toggle.
        """
        # First try to load the template
        template_content = self.load_template("components/controls/bb8-toggle.html")
        
        if template_content:
            # Set the checked state based on current theme
            is_dark = self.__theme_manager.is_dark_mode()
            if is_dark:
                template_content = template_content.replace('<input class="bb8-toggle__checkbox" type="checkbox">', 
                                                        '<input class="bb8-toggle__checkbox" type="checkbox" checked>')
            
            return template_content
        
        # Fallback to the default BB8 HTML
        html = BB8_HTML
        
        # Set the checked state based on current theme
        is_dark = self.__theme_manager.is_dark_mode()
        if is_dark:
            html = html.replace('<input class="bb8-toggle__checkbox" type="checkbox">', 
                            '<input class="bb8-toggle__checkbox" type="checkbox" checked>')
        
        return html

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> Dict[str, Any]:
        """Display the BB8 toggle component and handle theme changes.
        
        Returns:
            Dict containing theme information.
        """
        # Make sure the CSS is loaded
        self._load_bb8_toggle_css()
        
        # Render the HTML
        html = self.render()
        
        # Add JavaScript for handling the toggle change with a unique ID to avoid conflicts
        # Generate a unique ID for this toggle instance
        toggle_id = f"bb8_toggle_{self.__key}"
        
        js_code = f"""
        <script>
        (function() {{
            // Find the checkbox using its class, ensure we get the right one
            const checkbox = document.querySelector('.bb8-toggle__checkbox');
            if (!checkbox) {{
                console.error('BB8 checkbox not found');
                return;
            }}
            
            // Set a unique ID to make it easier to reference
            checkbox.id = "{toggle_id}";
            
            // Create a function to handle the theme change
            function handleThemeChange(event) {{
                const newTheme = event.target.checked ? 'dark' : 'light';
                console.log('BB8 toggle clicked, new theme:', newTheme);
                
                // Apply theme visually immediately for better UX
                document.documentElement.setAttribute('data-theme', newTheme);
                
                // Send the state to Streamlit
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: {{ 
                        theme: newTheme,
                        key: "{self.__key}",
                        toggled: event.target.checked
                    }}
                }}, '*');
            }}
            
            // Remove any existing listeners to avoid duplicates
            checkbox.removeEventListener('change', handleThemeChange);
            
            // Add the event listener
            checkbox.addEventListener('change', handleThemeChange);
            
            console.log('BB8 toggle initialized with ID: {toggle_id}');
        }})();
        </script>
        """
        
        # Combine HTML and JavaScript
        combined_html = f"{html}{js_code}"
        
        # Display the component
        st.markdown(combined_html, unsafe_allow_html=True)
        
        # Create a container for the toggle state if it doesn't exist
        if f"bb8_state_{self.__key}" not in st.session_state:
            st.session_state[f"bb8_state_{self.__key}"] = {
                "theme": self.__theme_manager.get_current_theme(),
                "toggled": self.__theme_manager.is_dark_mode()
            }
        
        # Check for theme state changes in session_state
        # This happens after the user clicks the toggle
        for key in st.session_state:
            if key == "theme" and st.session_state[key] != self.__theme_manager.get_current_theme():
                self._logger.debug(f"Theme changed in session_state: {st.session_state[key]}")
                self.__theme_manager.apply_theme(st.session_state[key])
                
                # Update our state tracker
                st.session_state[f"bb8_state_{self.__key}"]["theme"] = st.session_state[key]
                st.session_state[f"bb8_state_{self.__key}"]["toggled"] = (st.session_state[key] == "dark")
                
                # Call on_change callback if provided
                if self.__on_change:
                    self.__on_change(st.session_state[key])
                    
                # Force a page rerun to update all components
                st.rerun()
        
        # Return current state
        return {
            "theme": self.__theme_manager.get_current_theme(),
            "is_dark": self.__theme_manager.is_dark_mode(),
            "key": self.__key
        }