# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/base_view.py
# Description: Base view class for all application views
# Created: 2024-05-01

import streamlit as st
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, List, Union

from core.app_state.session_state import SessionState
from core.errors.ui_errors import UIError
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)

class BaseView(ABC):
    """Base view class that all views must inherit from.
    
    This abstract base class provides common functionality for all views,
    including error handling, state access, and a standard rendering interface.
    """
    
    def __init__(
        self,
        view_id: str,
        title: str,
        description: Optional[str] = None,
        icon: Optional[str] = None
    ):
        """Initialize the base view.
        
        Args:
            view_id: Unique identifier for the view
            title: Display title for the view
            description: Optional description of the view
            icon: Optional icon for the view (emoji or icon class)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Entering __init__ for view: {view_id}")
        
        self.view_id = view_id
        self.title = title
        self.description = description
        self.icon = icon
        self.theme_manager = ThemeManager()
        
        self.logger.debug(f"Exiting __init__ for view: {view_id}")
    
    def display(self) -> None:
        """Display the view with proper error handling.
        
        This method handles the complete view rendering process, including
        setup, rendering, and error handling.
        """
        self.logger.debug(f"Entering display for view: {self.view_id}")
        try:
            # Set up the view
            self._setup()
            
            # Display title with icon if present
            if self.icon:
                st.title(f"{self.icon} {self.title}")
            else:
                st.title(self.title)
            
            # Display description if present
            if self.description:
                st.write(self.description)
            
            # Render the view content
            self.render()
            self.logger.debug(f"View {self.view_id} rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Error displaying view {self.view_id}: {str(e)}", exc_info=True)
            self._handle_error(e)
            
        self.logger.debug(f"Exiting display for view: {self.view_id}")
    
    def _setup(self) -> None:
        """Set up the view before rendering.
        
        This method can be overridden by subclasses to perform initialization
        tasks before rendering.
        """
        self.logger.debug(f"Entering _setup for view: {self.view_id}")
        # Base implementation does nothing
        self.logger.debug(f"Exiting _setup for view: {self.view_id}")
        pass
    
    def _handle_error(self, error: Exception) -> None:
        """Handle errors that occur during view rendering.
        
        Args:
            error: The exception that was raised
        """
        self.logger.debug(f"Entering _handle_error for view: {self.view_id}")
        error_message = f"Error in {self.title} view: {str(error)}"
        self.logger.error(error_message, exc_info=True)
        
        # Display a user-friendly error message
        st.error(f"An error occurred while displaying this view: {str(error)}")
        
        # Show technical details in an expander for debugging
        with st.expander("Technical Details"):
            st.code(f"Error type: {type(error).__name__}\nError message: {str(error)}")
            
        self.logger.debug(f"Exiting _handle_error for view: {self.view_id}")
    
    @abstractmethod
    def render(self) -> None:
        """Render the view content.
        
        This method must be implemented by all subclasses to render
        the actual view content.
        """
        pass
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the session state.
        
        Args:
            key: The key to retrieve from session state
            default: Default value if key does not exist
            
        Returns:
            The value from session state or the default value
        """
        self.logger.debug(f"Entering get_state for key: {key}")
        value = SessionState.get(key, default)
        self.logger.debug(f"Exiting get_state for key: {key}")
        return value
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a value in the session state.
        
        Args:
            key: The key to set in session state
            value: The value to store
        """
        self.logger.debug(f"Entering set_state for key: {key}")
        try:
            SessionState.set(key, value)
            self.logger.debug(f"Successfully set state for key: {key}")
        except Exception as e:
            self.logger.error(f"Error setting state for key {key}: {str(e)}", exc_info=True)
            raise
        self.logger.debug(f"Exiting set_state for key: {key}")
    
    def has_state(self, key: str) -> bool:
        """Check if a key exists in the session state.
        
        Args:
            key: The key to check in session state
            
        Returns:
            True if the key exists, False otherwise
        """
        self.logger.debug(f"Entering has_state for key: {key}")
        result = SessionState.has_key(key)
        self.logger.debug(f"Exiting has_state for key: {key}, result: {result}")
        return result
    
    def get_view_data(self) -> Dict[str, Any]:
        """Get view data for rendering templates.
        
        Returns:
            Dict[str, Any]: View data dictionary
        """
        self.logger.debug(f"Entering get_view_data for view: {self.view_id}")
        try:
            data = {
                "view_id": self.view_id,
                "title": self.title,
                "description": self.description,
                "icon": self.icon,
                "is_dark_mode": self.theme_manager.is_dark_mode()
            }
            self.logger.debug(f"Exiting get_view_data for view: {self.view_id}")
            return data
        except Exception as e:
            self.logger.error(f"Error getting view data for {self.view_id}: {str(e)}", exc_info=True)
            # Return basic data to prevent failures
            return {
                "view_id": self.view_id,
                "title": self.title,
                "description": self.description,
                "icon": self.icon
            }
    
    def render_title(self) -> None:
        """Render the view title."""
        self.logger.debug(f"Entering render_title for view: {self.view_id}")
        try:
            if self.icon:
                title_html = f"{self.icon} {self.title}"
            else:
                title_html = self.title
            
            st.markdown(f"<h1 class='view-title'>{title_html}</h1>", unsafe_allow_html=True)
            
            if self.description:
                st.markdown(
                    f"<p class='view-description'>{self.description}</p>",
                    unsafe_allow_html=True
                )
            self.logger.debug(f"Title rendered successfully for view: {self.view_id}")
        except Exception as e:
            self.logger.error(f"Error rendering title for view {self.view_id}: {str(e)}", exc_info=True)
            # Fallback to simple title
            st.title(self.title)
        self.logger.debug(f"Exiting render_title for view: {self.view_id}")
    
    def safe_render(self) -> None:
        """Safely render the view with error handling."""
        self.logger.debug(f"Entering safe_render for view: {self.view_id}")
        try:
            self.render()
            self.logger.debug(f"View {self.view_id} rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering view {self.view_id}: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering this view: {str(e)}")
            
            # Emergency fallback
            st.markdown("""
            ## Something went wrong
            
            We encountered an error while trying to display this view.
            Please try refreshing the page or navigating to another view.
            """)
        self.logger.debug(f"Exiting safe_render for view: {self.view_id}")
    
    def display(self) -> None:
        """Display the view."""
        view_container_styles = """
        <style>
        .view-container-{} {{
            padding: 0.5rem;
        }}
        
        .view-title {{
            margin-bottom: 1rem;
            color: var(--text-color-primary);
        }}
        
        .view-description {{
            margin-bottom: 2rem;
            color: var(--text-color-secondary);
            font-size: 1.1rem;
        }}
        </style>
        """.format(self.view_id)
        
        st.markdown(view_container_styles, unsafe_allow_html=True)
        
        # Render title
        self.render_title()
        
        # Render content
        self.safe_render()
    
    def render_layout(self):
        """Render the view layout using Streamlit containers.
        
        Uses Streamlit's container hierarchy for better layout control.
        """
        # Use Streamlit containers for layout
        with st.container():
            # Header area
            self.render_header()
            
            # Main content
            with st.container():
                cols = st.columns([2, 5, 2])  # Example of column-based layout
                
                # Sidebar content in left column
                with cols[0]:
                    self.render_sidebar()
                    
                # Main content in center column
                with cols[1]:
                    self.render_content()
                    
                # Auxiliary content in right column
                with cols[2]:
                    self.render_auxiliary()
    
    def apply_common_layout(self):
        """Apply common layout styling to ensure consistency across views."""
        try:
            # Import and apply view styling utility if available
            try:
                from utils.ui.view_utils import apply_view_styling
                apply_view_styling()
            except ImportError:
                # Fallback to direct CSS injection if utility isn't available
                st.markdown("""
                <style>
                /* Fix content alignment */
                .block-container {
                    max-width: 100% !important;
                    padding-top: 1rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
                
                /* Make headers look better */
                h1, h2, h3 {
                    margin-bottom: 1rem !important;
                    margin-top: 0.5rem !important;
                    font-family: var(--font-primary, 'Poppins', sans-serif) !important;
                }
                
                /* Add space around elements */
                .stMarkdown {
                    margin-bottom: 0.5rem !important;
                }
                
                /* Remove empty columns */
                .stColumn:empty {
                    display: none !important;
                }
                </style>
                """, unsafe_allow_html=True)
        except Exception as e:
            self.logger.error(f"Error applying common layout: {str(e)}") 