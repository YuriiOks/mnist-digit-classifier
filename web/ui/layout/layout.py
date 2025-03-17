# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/layout.py
# Description: Layout component for the application
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from ui.components.base.component import Component
from ui.theme.theme_manager import theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils


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
        self.header = Header(title=title, actions_html=header_actions)
        self.footer = Footer(content=footer_content)
        self.sidebar = Sidebar()
        
        # Ensure theme is initialized
        theme_manager.initialize()
    
    def render_header(self) -> None:
        """Render just the application header."""
        # Display header
        self.header.display()

    def render_footer(self) -> None:
        """Render just the application footer."""
        # Add spacing before footer
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        self.footer.display()
    
    def render(self) -> None:
        """
        Render the full application layout.
        This is maintained for backward compatibility.
        """
        # Display sidebar
        self.sidebar.display()
        
        # Display header
        self.header.display()
        
        # Main content container will be added by the individual view
        
        # Display footer (with spacing)
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        self.footer.display()