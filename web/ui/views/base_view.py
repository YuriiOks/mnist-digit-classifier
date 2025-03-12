# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/base_view.py
# Description: Base view class for all application views
# Created: 2024-05-01

import streamlit as st
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional

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
        title: Optional[str] = None,
        description: Optional[str] = None,
        icon: Optional[str] = None
    ):
        """Initialize the base view.
        
        Args:
            view_id: Unique identifier for the view
            title: Display title for the view (not rendered by default)
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
        
        # Initialize state
        self._setup()
        
        self.logger.debug(f"Exiting __init__ for view: {view_id}")
    
    def _setup(self) -> None:
        """Set up the view (can be overridden by subclasses)."""
        pass
    
    def display(self) -> None:
        """Display the view (calls render internally)."""
        self.logger.debug(f"Entering display for view: {view_id}")
        try:
            # Don't render title automatically - let views handle their own headers
            # Instead, just call render directly
            self.render()
        except Exception as e:
            self.logger.error(f"Error displaying view: {str(e)}", exc_info=True)
            self._handle_error(e)
        self.logger.debug(f"Exiting display for view: {view_id}")
    
    @abstractmethod
    def render(self) -> None:
        """Render the view content (must be implemented by subclasses)."""
        pass
    
    def get_view_data(self) -> Dict[str, Any]:
        """Get view-specific data for templates.
        
        Returns:
            Dict[str, Any]: Dictionary of view data
        """
        return {
            "view_id": self.view_id,
            "title": self.title,
            "description": self.description,
            "icon": self.icon
        }
    
    def _handle_error(self, error: Exception) -> None:
        """Handle view rendering errors.
        
        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Error in view {self.view_id}: {str(error)}", exc_info=True)
        
        if isinstance(error, UIError):
            # UI errors are displayed with their specific message
            st.error(f"UI Error: {str(error)}")
        else:
            # Generic errors get a more user-friendly message
            st.error("An error occurred while displaying this view.")
            st.error(f"Details: {str(error)}")
            
        # Display a help message
        st.info("Try reloading the page or contact support if the issue persists.")
    
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
    
    def render_title(self) -> None:
        """Render the view title (hidden by CSS)."""
        self.logger.debug(f"Entering render_title for view: {self.view_id}")
        try:
            if self.icon:
                title_html = f"{self.icon} {self.title}"
            else:
                title_html = self.title
            
            # Only render the title - completely skip the description
            st.markdown(f"<h1 class='view-title'>{title_html}</h1>", unsafe_allow_html=True)
            
            self.logger.debug(f"Title rendered successfully for view: {self.view_id}")
        except Exception as e:
            self.logger.error(f"Error rendering title for view {self.view_id}: {str(e)}", exc_info=True)
            # Fallback to simple title - no description
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
        """Display the view (calls render internally)."""
        self.logger.debug(f"Entering display for view: {self.view_id}")    
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
            # Apply view styling first
            try:
                logger.info("Applying view styling")
                from utils.ui.view_utils import apply_view_styling
                logger.info("View styling utility found")
                apply_view_styling()
            except ImportError: 
                logger.warning("View styling utility not found")
                
            # Now apply component-specific CSS
            self._apply_css()
            
        except Exception as e:
            self.logger.error(f"Error applying common layout: {str(e)}")

    def _apply_css(self, css_modules=None):
        """Apply CSS for view components.
        
        Args:
            css_modules: List of CSS module names to apply, defaults to button_css and card_css
        """
        if css_modules is None:
            css_modules = ["button_css", "card_css"]
            
        self.logger.debug(f"Applying CSS modules: {css_modules}")
        
        for css_module in css_modules:
            try:
                # First try relative import (from current module)
                module = __import__(f"utils.css.{css_module}", fromlist=[""])
                loader_func = getattr(module, f"load_{css_module}")
                loader_func()
                self.logger.debug(f"Applied CSS module: {css_module}")
                
            except (ImportError, AttributeError, ValueError) as e:
                self.logger.debug(f"First import attempt failed for {css_module}: {str(e)}")
                try:
                    # Try alternate import path
                    if css_module == "button_css":
                        from utils.css.button_css import load_button_css
                        load_button_css()
                    elif css_module == "card_css":
                        from utils.css.card_css import load_card_css
                        load_card_css()
                    else:
                        # Dynamic import for other modules
                        module_name = f"utils.css.{css_module}"
                        module = __import__(module_name, fromlist=[""])
                        getattr(module, f"load_{css_module}")()
                        
                    self.logger.debug(f"Applied CSS module: {css_module}")
                    
                except (ImportError, AttributeError) as e:
                    self.logger.warning(f"Could not load CSS module '{css_module}': {str(e)}") 