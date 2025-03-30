# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/layout/layout.py
# Description: Layout utility class with improved debugging
# Created: 2025-03-17
# Updated: 2025-03-30

import streamlit as st
import logging
from typing import Optional, Dict, Any

# Import the individual layout components
from ui.layout.header import Header
from ui.layout.footer import Footer
from ui.layout.sidebar import Sidebar
from ui.theme.theme_manager import theme_manager
from utils.aspects import AspectUtils


class Layout:
    """Utility class for managing application layout."""

    def __init__(
        self,
        title: str = "MNIST Digit Classifier",
        header_actions: str = "",
        footer_content: Optional[str] = None,
    ):
        """
        Initialize the layout manager.

        Args:
            title: Application title.
            header_actions: HTML for additional header actions.
            footer_content: Custom footer content (HTML).
        """
        self.__logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.__logger.info("Initializing Layout manager")

        # Initialize components with proper error handling
        self.__header = self.__init_header(title, header_actions)
        self.__footer = self.__init_footer(footer_content)
        self.__sidebar = self.__init_sidebar()

        # Ensure theme is initialized
        theme_manager.initialize()

    @property
    def header(self) -> Optional[Header]:
        """Get the header component."""
        return self.__header

    @property
    def footer(self) -> Optional[Footer]:
        """Get the footer component."""
        return self.__footer

    @property
    def sidebar(self) -> Optional[Sidebar]:
        """Get the sidebar component."""
        return self.__sidebar

    def __init_header(self, title: str, actions_html: str) -> Optional[Header]:
        """Initialize header component with error handling."""
        try:
            header = Header(title=title, actions_html=actions_html)
            self.__logger.info("Header component initialized")
            return header
        except Exception as e:
            self.__logger.error(f"Error initializing Header: {str(e)}", exc_info=True)
            return None

    def __init_footer(self, content: Optional[str]) -> Optional[Footer]:
        """Initialize footer component with error handling."""
        try:
            footer = Footer(content=content)
            self.__logger.info("Footer component initialized")
            return footer
        except Exception as e:
            self.__logger.error(f"Error initializing Footer: {str(e)}", exc_info=True)
            return None

    def __init_sidebar(self) -> Optional[Sidebar]:
        """Initialize sidebar component with error handling."""
        try:
            sidebar = Sidebar()
            self.__logger.info("Sidebar component initialized")
            return sidebar
        except Exception as e:
            self.__logger.error(f"Error initializing Sidebar: {str(e)}", exc_info=True)
            return None

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render_header(self) -> None:
        """Render just the application header."""
        if self.__header is None:
            self.__logger.error("Cannot render header: component is not initialized")
            st.error("Error initializing header component")
            return

        self.__header.display()

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render_footer(self) -> None:
        """Render just the application footer."""
        if self.__footer is None:
            self.__logger.error("Cannot render footer: component is not initialized")
            st.error("Error initializing footer component")
            return

        # Add spacing before footer
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        self.__footer.display()

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render_sidebar(self) -> None:
        """Render just the application sidebar."""
        if self.__sidebar is None:
            self.__logger.error("Cannot render sidebar: component is not initialized")
            st.error("Error initializing sidebar component")
            return

        self.__sidebar.display()

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> None:
        """
        Render the full application layout.
        This is maintained for backward compatibility.
        """
        # Display sidebar
        self.render_sidebar()

        # Display header
        self.render_header()

        # Main content container will be added by the individual view

        # Display footer (with spacing)
        self.render_footer()
