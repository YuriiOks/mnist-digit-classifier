# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/layout.py
# Description: Layout utility class with improved debugging
# Created: 2025-03-17

import streamlit as st
import logging
import traceback
from typing import Optional

# Import the individual layout components
from ui.layout.header import Header
from ui.layout.footer import Footer
from ui.layout.sidebar import Sidebar
from ui.theme.theme_manager import theme_manager


class Layout:
    """Utility class for managing application layout."""
    
    def __init__(
        self,
        title: str = "MNIST Digit Classifier",
        header_actions: str = "",
        footer_content: Optional[str] = None
    ):
        """
        Initialize the layout manager.
        
        Args:
            title: Application title.
            header_actions: HTML for additional header actions.
            footer_content: Custom footer content (HTML).
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing Layout manager")
        
        try:
            self.header = Header(title=title, actions_html=header_actions)
            self.logger.info("Header component initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Header: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.header = None
            
        try:
            self.footer = Footer(content=footer_content)
            self.logger.info("Footer component initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Footer: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.footer = None
            
        try:
            self.sidebar = Sidebar()
            self.logger.info("Sidebar component initialized")
        except Exception as e:
            self.logger.error(f"Error initializing Sidebar: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.sidebar = None
        
        # Ensure theme is initialized
        theme_manager.initialize()
    
    def render_header(self) -> None:
        """Render just the application header."""
        if self.header is None:
            self.logger.error("Cannot render header: component is not initialized")
            st.error("Error initializing header component")
            return
            
        try:
            self.logger.info("Rendering header")
            self.header.display()
            self.logger.info("Header rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering header: {str(e)}")
            self.logger.error(traceback.format_exc())
            st.error("Error rendering header component")

    def render_footer(self) -> None:
        """Render just the application footer."""
        if self.footer is None:
            self.logger.error("Cannot render footer: component is not initialized")
            st.error("Error initializing footer component")
            return
            
        try:
            # Add spacing before footer
            st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
            self.logger.info("Rendering footer")
            self.footer.display()
            self.logger.info("Footer rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering footer: {str(e)}")
            self.logger.error(traceback.format_exc())
            st.error("Error rendering footer component")
    
    def render_sidebar(self) -> None:
        """Render just the application sidebar."""
        if self.sidebar is None:
            self.logger.error("Cannot render sidebar: component is not initialized")
            st.error("Error initializing sidebar component") 
            return
            
        try:
            self.logger.info("Rendering sidebar")
            self.sidebar.display()
            self.logger.info("Sidebar rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering sidebar: {str(e)}")
            self.logger.error(traceback.format_exc())
            st.error("Error rendering sidebar component")
    
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