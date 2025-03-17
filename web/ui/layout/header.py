# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/header.py
# Description: Header component for the application
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any, Callable

from ui.components.base.component import Component
from ui.theme.theme_manager import theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils


class Header(Component[None]):
    """Header component for the application."""
    
    def __init__(
        self,
        title: str = "MNIST Digit Classifier",
        actions_html: str = "",
        toggle_theme_callback: Optional[Callable] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the header component.
        
        Args:
            title: Application title to display.
            actions_html: HTML for additional actions in the header.
            toggle_theme_callback: Function to call when theme toggle is clicked.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
            key: Unique key for Streamlit.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_type="layout",
            component_name="header",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )
        self.__title = title
        self.__actions_html = actions_html
        self.__toggle_theme_callback = toggle_theme_callback or theme_manager.toggle_theme
    
    @property
    def title(self) -> str:
        """Get the header title."""
        return self.__title
    
    @title.setter
    def title(self, value: str) -> None:
        """Set the header title."""
        self.__title = value
    
    @property
    def actions_html(self) -> str:
        """Get the header actions HTML."""
        return self.__actions_html
    
    @actions_html.setter
    def actions_html(self, value: str) -> None:
        """Set the header actions HTML."""
        self.__actions_html = value
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the header component.
        
        Returns:
            HTML representation of the header.
        """
        # Try to render using template
        template_content = self.render_template(
            "components/layout/header.html",
            {
                "TITLE": self.__title,
                "ACTIONS_HTML": self.__actions_html
            }
        )
        
        if template_content:
            return template_content
        
        # Fallback to direct HTML generation
        return f"""
        <div class="app-header">
            <h1>{self.__title}</h1>
            {self.__actions_html}
        </div>
        """
    
    @AspectUtils.catch_errors
    def display(self) -> None:
        """Display the header component in Streamlit."""
        # Render the HTML
        header_html = self.render()
        
        # Display in Streamlit
        st.markdown(header_html, unsafe_allow_html=True)
