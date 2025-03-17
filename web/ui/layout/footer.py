# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/footer.py
# Description: Footer component for the application
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ui.components.base.component import Component
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils


class Footer(Component[None]):
    """Footer component for the application."""
    
    def __init__(
        self,
        content: Optional[str] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the footer component.
        
        Args:
            content: Footer content (HTML).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dict of HTML attributes to apply to the component.
            key: Unique key for Streamlit.
            **kwargs: Additional keyword arguments.
        """
        # Initialize parent with explicitly named parameters
        super().__init__(
            component_type="layout",
            component_name="footer",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key or "app_footer",
            **kwargs
        )
        
        # Set default content if none provided
        self.__content = content or self.__get_default_content()
        self._logger.debug("Footer component initialized successfully")
    
    @property
    def content(self) -> str:
        """Get the footer content."""
        return self.__content
    
    @content.setter
    def content(self, value: str) -> None:
        """Set the footer content."""
        self.__content = value
    
    def __get_default_content(self) -> str:
        """
        Get default footer content.
        
        Returns:
            Default footer content with current year.
        """
        current_year = datetime.now().year
        return f"""
        © {current_year} MNIST Digit Classifier | 
        Developed by <a href="https://github.com/YuriODev" target="_blank">YuriODev</a> | 
        <a href="https://github.com/YuriiOks/mnist-digit-classifier" target="_blank">
        <span style="white-space: nowrap;">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" 
        viewBox="0 0 16 16" style="vertical-align: text-bottom; margin-right: 4px;">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-
        .28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-
        .87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-
        2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 
        2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-
        3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 
        8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
        </svg>GitHub</span></a>
        """
    
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> str:
        """
        Render the footer component.
        
        Returns:
            HTML representation of the footer.
        """
        # Try to render using template
        template_content = self.render_template(
            "components/layout/footer.html",
            {
                "CONTENT": self.__content,
                "YEAR": datetime.now().year
            }
        )
        
        if template_content:
            return template_content
        
        # Fallback to direct HTML generation
        return f"""
        <footer class="app-footer">
            <div class="footer-content">
                {self.__content}
            </div>
        </footer>
        """
    
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> None:
        """Display the footer component in Streamlit."""
        # Load CSS for footer
        self._load_component_css()
        
        # Render the HTML
        footer_html = self.render()
        
        # Display in Streamlit
        st.markdown(footer_html, unsafe_allow_html=True)
    
    def _load_component_css(self) -> None:
        """Load CSS specific to this component."""
        css_path = "components/layout/footer.css"
        css_content = resource_manager.load_css(css_path)
        if css_content:
            resource_manager.inject_css(css_content)
            self._logger.debug(f"Loaded CSS: {css_path}")
        else:
            self._logger.warning(f"Could not load CSS: {css_path}")