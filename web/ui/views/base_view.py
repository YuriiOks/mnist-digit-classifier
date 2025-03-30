# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/views/base_view.py
# Description: Base view class for application views
# Created: 2025-03-17

import streamlit as st
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from utils.resource_manager import resource_manager
from ui.theme.theme_manager import theme_manager


class View(ABC):
    """Base class for all application views."""

    def __init__(
        self, name: str, title: str, description: Optional[str] = None
    ):
        """
        Initialize a view.

        Args:
            name: View name/identifier
            title: View title for display
            description: Optional view description
        """
        self.name = name
        self.title = title
        self.description = description
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def load_view_css(self) -> None:
        """Load CSS specific to this view."""
        # Try to load view-specific CSS
        resource_manager.load_and_inject_css(
            [f"views/{self.name}.css", "views/view_styles.css"]
        )

    @abstractmethod
    def render(self) -> None:
        """Render the view content. Must be implemented by subclasses."""
        pass

    def pre_render(self) -> None:
        """Perform setup before rendering the view."""
        # Load view-specific CSS
        self.load_view_css()

        # Add view title and description
        if hasattr(self, "show_header") and self.show_header:
            if self.title:
                st.markdown(f"<h2>{self.title}</h2>", unsafe_allow_html=True)
            if self.description:
                st.markdown(
                    f"<p>{self.description}</p>", unsafe_allow_html=True
                )

    def post_render(self) -> None:
        """Perform cleanup after rendering the view."""
        pass

    def display(self) -> None:
        """Display the view with pre and post processing."""
        try:
            self.pre_render()
            self.render()
            self.post_render()
        except Exception as e:
            self.logger.error(
                f"Error rendering view '{self.name}': {str(e)}", exc_info=True
            )
            st.error(
                f"An error occurred while rendering the {self.title} view."
            )
            if st.session_state.get("debug_mode", False):
                st.exception(e)
