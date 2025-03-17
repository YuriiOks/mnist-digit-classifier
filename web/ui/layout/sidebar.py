# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/sidebar.py
# Description: Sidebar component for the application
# Created: 2025-03-17

import streamlit as st
import logging
from typing import Optional, List, Dict
from datetime import datetime

from ui.components.base.component import Component
from ui.theme.theme_manager import theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils
from ui.components.controls.bb8_toggle import BB8Toggle
from core.app_state.navigation_state import NavigationState


class Sidebar(Component[None]):
    """Sidebar navigation component."""
    
    def __init__(
        self,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the sidebar component."""
        super().__init__(
            component_type="layout",
            component_name="sidebar",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key or "app_sidebar",
            **kwargs
        )
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the sidebar component to HTML.
        
        Returns:
            HTML representation of the sidebar.
        """
        # This is a minimal implementation to satisfy the abstract method
        # We're primarily using Streamlit's built-in sidebar rather than custom HTML
        current_year = datetime.now().year
        version = "1.0.0"
        
        return f"""
        <div class="sidebar-container">
            <div class="sidebar-header">
                <div class="gradient-text">MNIST App</div>
                <div class="sidebar-subheader">Digit Classification AI</div>
            </div>
            
            <div class="nav-buttons-container">
                <!-- Navigation buttons rendered by Streamlit -->
            </div>
            
            <div class="divider"></div>
            
            <div class="toggle-container">
                <!-- Theme toggle rendered by Streamlit -->
            </div>
            
            <div class="divider"></div>
            
            <div class="sidebar-footer">
                <p>Version {version}</p>
                <p>© {current_year} MNIST Classifier</p>
            </div>
        </div>
        """
    
    @AspectUtils.catch_errors
    def display(self) -> None:
        """Display the sidebar component in Streamlit."""
        # Load sidebar CSS
        sidebar_css = resource_manager.load_css("components/layout/sidebar.css")
        buttons_css = resource_manager.load_css("components/controls/buttons.css")
        
        # Combine and inject CSS
        if sidebar_css or buttons_css:
            combined_css = ""
            if sidebar_css:
                combined_css += sidebar_css
            if buttons_css:
                combined_css += buttons_css
            resource_manager.inject_css(combined_css)
        
        # Use Streamlit's sidebar container
        with st.sidebar:
            # Apply theme CSS variables
            theme_data = theme_manager.get_theme_data()
            theme_manager._apply_css_variables(theme_data)
            
            # Header
            st.markdown(
                """
                <div class="sidebar-header">
                    <div class="gradient-text">MNIST App</div>
                    <div class="sidebar-subheader">Digit Classification AI</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Navigation items
            nav_items = NavigationState.get_routes()
            active_view = NavigationState.get_active_view()
            
            # Create navigation buttons
            for item in nav_items:
                if st.button(
                    f"{item['icon']} {item['label']}",
                    key=f"nav_{item['id']}",
                    type="primary" if active_view == item['id'] else "secondary",
                    use_container_width=True
                ):
                    NavigationState.set_active_view(item['id'])
                    st.rerun()
            
            # Add JavaScript to style active button
            script = f"""
            <script>
                (function() {{
                    setTimeout(function() {{
                        const activeViewId = '{active_view}';
                        const activeButton = document.querySelector(`[data-testid="stButton"] button[key="nav_${{activeViewId}}"]`);
                        if (activeButton) {{
                            activeButton.classList.add('nav-active');
                        }}
                    }}, 100);
                }})();
            </script>
            """
            st.markdown(script, unsafe_allow_html=True)
            
            # Divider
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # BB8 Toggle
            with st.container():
                bb8_toggle = BB8Toggle(
                    theme_manager_instance=theme_manager,
                    on_change=lambda new_theme: st.rerun(),
                    key="sidebar_bb8_toggle"
                )
                bb8_toggle.display()
            
            # Divider
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Footer
            current_year = datetime.now().year
            version = "1.0.0"
            
            st.markdown(
                f"""
                <div class="sidebar-footer">
                    <p>Version {version}</p>
                    <p>© {current_year} MNIST Classifier</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Handle theme toggle callback
            if "theme" in st.session_state:
                requested_theme = st.session_state["theme"]
                current_theme = theme_manager.get_current_theme()
                
                if requested_theme != current_theme:
                    theme_manager.apply_theme(requested_theme)
                    st.rerun()
            
            # Handle navigation callback
            if "view" in st.session_state:
                requested_view = st.session_state["view"]
                current_view = NavigationState.get_active_view()
                
                if requested_view != current_view:
                    NavigationState.set_active_view(requested_view)
                    st.rerun()