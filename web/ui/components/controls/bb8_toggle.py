# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/bb8_toggle.py
# Description: BB8 theme toggle component
# Created: 2024-05-03

import streamlit as st
from typing import Callable, Optional
import logging
from pathlib import Path

from ui.theme.theme_manager import ThemeManager
from utils.template_loader import TemplateLoader

logger = logging.getLogger(__name__)


class BB8Toggle:
    """A BB8-themed toggle component for switching between light and dark themes."""

    def __init__(self, theme_manager: ThemeManager, on_change: Optional[Callable] = None):
        self.theme_manager = theme_manager
        self.on_change = on_change
        self.template_loader = TemplateLoader()
        
        # Initialize session state for theme if not already present
        if "current_theme" not in st.session_state:
            st.session_state.current_theme = "light"

    def _load_css(self):
        """Load the BB8 toggle CSS file."""
        css_path = Path(__file__).parent.parent.parent.parent / "assets" / "css" / "components" / "controls" / "bb8-toggle.css"
        
        if css_path.exists():
            with open(css_path, "r") as f:
                return f.read()
        else:
            logger.error(f"BB8 toggle CSS file not found at {css_path}")
            return ""
            
    def render(self):
        """Render the BB8 theme toggle."""
        # Get the current theme to set the initial state of the checkbox
        current_theme = st.session_state.current_theme
        is_dark_mode = current_theme == "dark"
        
        try:
            # Load CSS
            css_code = self._load_css()
            
            # Load HTML template
            template = self.template_loader.load_template("components/controls/bb8-toggle.html")
            
            if not template:
                logger.error("Failed to load BB8 toggle HTML template")
                # Fallback to a basic toggle if template not found
                st.select_slider("Theme", ["Light", "Dark"], 
                                value="Dark" if is_dark_mode else "Light",
                                key="theme_fallback_toggle")
                return
                
            # Replace the checkbox state placeholder in the template
            html_code = template.replace('type="checkbox"', 
                                        f'type="checkbox" {"checked" if is_dark_mode else ""}')
            
            # Combine HTML and CSS, and use st.markdown to render
            st.markdown(f"<style>{css_code}</style>{html_code}", unsafe_allow_html=True)
            
            # Handle checkbox change (using JavaScript)
            st.markdown(
                """
                <script>
                // Find all BB8 toggle checkboxes
                document.addEventListener('DOMContentLoaded', function() {
                    const toggles = document.querySelectorAll('.bb8-toggle__checkbox');
                    
                    // Function to handle theme change
                    function handleThemeChange(checkbox) {
                        if (checkbox.checked) {
                            document.documentElement.setAttribute('data-theme', 'dark');
                            // Call Streamlit function via parent window
                            window.parent.callToggleTheme('dark');
                        } else {
                            document.documentElement.setAttribute('data-theme', 'light');
                            window.parent.callToggleTheme('light');
                        }
                    }
                    
                    // Attach event listener to each toggle
                    toggles.forEach(toggle => {
                        toggle.addEventListener('change', () => handleThemeChange(toggle));
                    });
                });
                </script>
                """,
                unsafe_allow_html=True,
            )
            
            logger.debug(f"BB8 Toggle rendered. Current theme: {current_theme}")
            
        except Exception as e:
            logger.error(f"Error rendering BB8 toggle: {str(e)}", exc_info=True)
            # Fallback to a basic toggle
            st.error("Error loading theme toggle component")
            st.select_slider("Theme", ["Light", "Dark"], 
                            value="Dark" if is_dark_mode else "Light",
                            key="theme_fallback_toggle")
            
        # Call the on_change callback if provided
        if self.on_change:
            self.on_change(current_theme)