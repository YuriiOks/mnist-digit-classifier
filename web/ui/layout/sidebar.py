# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/sidebar.py

import streamlit as st
from typing import Dict, Any, Optional, List, Callable
import logging
import uuid

from ui.components.base.component import Component
from core.app_state.navigation_state import NavigationState
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)

class Sidebar(Component):
    """Sidebar layout component for the application."""
    
    def __init__(
        self,
        title: str = "Navigation",
        footer_text: Optional[str] = None
    ):
        """Initialize the sidebar component."""
        logger.debug(f"Initializing Sidebar with title: {title}")
        super().__init__(component_type="layout", component_name="sidebar")
        self.title = title
        self.footer_text = footer_text
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def display(self) -> None:
        """Display the sidebar component."""
        try:
            self.logger.debug("Rendering sidebar")
            
            with st.sidebar:
                # Display sidebar title
                self.logger.debug(f"Displaying sidebar title: {self.title}")
                st.markdown(f"## {self.title}")
                
                # Get navigation routes
                routes = NavigationState.get_routes()
                active_view = NavigationState.get_active_view()
                self.logger.debug(f"Retrieved {len(routes)} navigation routes, active view: {active_view}")
                
                # Create simple navigation using Streamlit components
                for route in routes:
                    item_id = route.get("id")
                    label = route.get("label", "")
                    icon = route.get("icon", "")
                    
                    # Generate a unique key for each button
                    # This prevents conflicts with session state keys
                    button_key = f"nav_btn_{item_id}_{uuid.uuid4().hex[:4]}"
                    self.logger.debug(f"Creating navigation button for {item_id} with key {button_key}")
                    
                    # Create a clickable button for each navigation item
                    if st.button(
                        f"{icon} {label}", 
                        key=button_key,
                        use_container_width=True,
                        type="primary" if item_id == active_view else "secondary"
                    ):
                        self.logger.info(f"Navigation button clicked: {item_id}")
                        NavigationState.set_active_view(item_id)
                        self.logger.debug("Triggering app rerun to apply navigation change")
                        st.rerun()
                
                # Add spacer
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                # Display footer if provided
                if self.footer_text:
                    self.logger.debug(f"Displaying sidebar footer text: {self.footer_text}")
                    st.markdown(
                        f"<div style='position: fixed; bottom: 20px; width: 100%;'>{self.footer_text}</div>", 
                        unsafe_allow_html=True
                    )
            
            self.logger.debug("Sidebar rendered successfully")
        except Exception as e:
            self.logger.error(f"Error displaying sidebar: {str(e)}", exc_info=True)
            st.sidebar.error(f"Error loading navigation: {str(e)}")
    
    def get_template_variables(self) -> dict:
        """Get template variables for rendering.
        
        Returns:
            dict: Template variables
        """
        self.logger.debug("Getting template variables for sidebar")
        try:
            variables = super().get_template_variables()
            
            # Get navigation items
            routes = NavigationState.get_routes()
            active_view = NavigationState.get_active_view()
            self.logger.debug(f"Preparing navigation items: {len(routes)} routes, active: {active_view}")
            
            # Format for template
            sidebar_items = ""
            for route in routes:
                item_id = route.get("id", "")
                label = route.get("label", "")
                icon = route.get("icon", "")
                is_active = item_id == active_view
                
                # Add to sidebar items HTML
                sidebar_items += f"""
                <li class="sidebar-item {('active' if is_active else '')}">
                    <a href="javascript:void(0);" data-view="{item_id}">{icon} {label}</a>
                </li>
                """
            
            variables.update({
                "SIDEBAR_TITLE": self.title,
                "SIDEBAR_ITEMS": sidebar_items,
                "SIDEBAR_FOOTER": self.footer_text or ""
            })
            
            self.logger.debug("Template variables prepared successfully")
            return variables
        except Exception as e:
            self.logger.error(f"Error getting template variables: {str(e)}", exc_info=True)
            # Return basic variables to prevent rendering failure
            return {
                "SIDEBAR_TITLE": self.title,
                "SIDEBAR_ITEMS": "",
                "SIDEBAR_FOOTER": self.footer_text or ""
            }