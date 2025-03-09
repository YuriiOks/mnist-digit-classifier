# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/header.py
# Description: Application header component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Callable

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class Header(Component):
    """Header component for the application.
    
    This component renders the application header with title and theme toggle.
    """
    
    def __init__(
        self,
        title: str = "MNIST Digit Classifier",
        actions_html: str = "",
        toggle_theme_callback: Optional[Callable] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize the header component.
        
        Args:
            title: Application title to display.
            actions_html: HTML for additional actions in the header.
            toggle_theme_callback: Function to call when theme toggle is clicked.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the 
                component.
        """
        super().__init__(
            component_type="layout",
            component_name="header",
            id=id,
            classes=classes,
            attributes=attributes
        )
        self.title = title
        self.actions_html = actions_html
        if toggle_theme_callback:
            self.toggle_theme_callback = toggle_theme_callback
        else:
            self.toggle_theme_callback = ThemeManager.toggle_theme
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    def display(self) -> None:
        """Display the header component."""
        self.logger.debug("Displaying header component")
        try:
            # Create a simplified header with gradient background and just text
            header_css = """
            <style>
            /* Header with gradient background */
            .app-header {
                background: linear-gradient(
                    90deg,
                    var(--color-primary-light, #6366F1),
                    var(--color-secondary, #06B6D4)
                );
                color: white;
                padding: 1rem 2rem;
                margin-bottom: 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            /* Header title styling with text shadow for better readability */
            .app-header h1 {
                margin: 0;
                font-size: 2rem;
                font-weight: 600;
                font-family: var(--font-primary, 'Poppins', sans-serif);
                color: white;
                letter-spacing: 0.5px;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                text-align: center;
            }
            
            /* Add subtle shimmer effect */
            .app-header::after {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(
                    to right,
                    rgba(255, 255, 255, 0) 0%,
                    rgba(255, 255, 255, 0.3) 50%,
                    rgba(255, 255, 255, 0) 100%
                );
                transform: rotate(30deg);
                animation: headerShine 6s infinite linear;
                pointer-events: none;
            }
            
            @keyframes headerShine {
                0% {
                    transform: rotate(30deg) translate(-100%, -100%);
                }
                100% {
                    transform: rotate(30deg) translate(100%, 100%);
                }
            }
            </style>
            """
            
            # Simple header with just the title
            header_html = f"""
            <div class="app-header">
                <h1>{self.title}</h1>
            </div>
            """
            
            # Render the header
            st.markdown(header_css + header_html, unsafe_allow_html=True)
            
            self.logger.debug("Header displayed successfully")
        except Exception as e:
            self.logger.error(
                f"Error displaying header: {str(e)}", 
                exc_info=True
            )
            st.error("Error loading header component")

    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for header rendering."""
        try:
            variables = super().get_template_variables()
            variables.update({
                "TITLE": self.title,
                "ACTIONS_HTML": self.actions_html,
                "TOGGLE_THEME_JS": "toggleTheme()"
            })
            return variables
        except Exception as e:
            self.logger.error(
                f"Error getting template variables: {str(e)}", 
                exc_info=True
            )
            # Return basic variables to prevent rendering failure
            return {
                "TITLE": self.title,
                "ACTIONS_HTML": "",
                "TOGGLE_THEME_JS": "void(0)"
            }
    
    @staticmethod
    def toggle_theme_callback():
        """Toggle between light and dark theme."""
        ThemeManager.toggle_theme()
    
    def add_theme_toggle_callback(self) -> None:
        """Add JavaScript callback for theme toggle button."""
        # Add click handler for theme toggle
        if st.button("Toggle Theme", key="theme_toggle_btn", type="secondary"):
            self.toggle_theme_callback()
            # Rerun the app to apply the new theme
            st.rerun() 
