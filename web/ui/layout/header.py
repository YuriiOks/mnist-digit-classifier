# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/layout/header.py
# Description: Header component for the application
# Created: 2025-03-16
# Updated: 2025-03-30

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
        **kwargs,
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
        # Set basic properties before parent initialization
        self.__title = title
        self.__actions_html = actions_html
        self.__toggle_theme_callback = (
            toggle_theme_callback or theme_manager.toggle_theme
        )

        # Initialize parent with explicitly named parameters
        super().__init__(
            component_type="layout",
            component_name="header",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key or "app_header",
            **kwargs,
        )

        self._logger.debug("Header component initialized successfully")

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
    @AspectUtils.log_method
    def render(self) -> str:
        """
        Render the header component.

        Returns:
            HTML representation of the header.
        """
        # Try to render using template
        template_content = self.render_template(
            "components/layout/header.html",
            {"TITLE": self.__title, "ACTIONS_HTML": self.__actions_html},
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
    @AspectUtils.log_method
    def display(self) -> None:
        """Display the header component in Streamlit."""
        # Load CSS for header
        self._load_component_css()

        # Render the HTML
        header_html = self.render()

        # Display in Streamlit
        st.markdown(header_html, unsafe_allow_html=True)

    def _load_component_css(self) -> None:
        """Load CSS specific to this component."""
        css_path = "components/layout/header.css"
        css_content = resource_manager.load_css(css_path)
        if css_content:
            resource_manager.inject_css(css_content)
            self._logger.debug(f"Loaded CSS: {css_path}")
        else:
            self._logger.warning(f"Could not load CSS: {css_path}")
