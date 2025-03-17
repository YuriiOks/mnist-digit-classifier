# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/sidebar.py
# Description: Sidebar component for the application
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager, theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils
from ui.components.controls.bb8_toggle import BB8Toggle


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
        """
        Initialize the sidebar component.
        
        Args:
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dict of HTML attributes to apply to the component.
            key: Unique key for Streamlit.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_type="layout",
            component_name="sidebar",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )
    
    @AspectUtils.catch_errors
    def render_navigation_buttons(self) -> str:
        """
        Render the navigation buttons.
        
        Returns:
            HTML for navigation buttons.
        """
        # Navigation items
        from core.app_state.navigation_state import NavigationState
        nav_items = NavigationState.get_routes()
        
        # Get active view
        active_view = NavigationState.get_active_view()
        
        buttons_html = ""
        for item in nav_items:
            view_id = item["id"]
            label = f"{item['icon']} {item['label']}"
            active_class = "active" if view_id == active_view else ""
            buttons_html += f"""
            <div data-testid="stButton" data-baseweb-key="nav_{view_id}_btn">
                <button 
                    type="button" 
                    class="{active_class}"
                    onclick="window.parent.postMessage(
                        {{
                            type: 'streamlit:setComponentValue',
                            value: {{ view: '{view_id}' }}
                        }}, '*'
                    )"
                >
                    {label}
                </button>
            </div>
            """
        
        return buttons_html
    
    @AspectUtils.catch_errors
    def render_theme_toggle(self) -> str:
        """
        Render the theme toggle.
        
        Returns:
            HTML for theme toggle component.
        """
        # Try to load BB8 toggle component
        bb8_html = resource_manager.load_template("components/controls/bb8-toggle.html")
        if bb8_html:
            is_dark = theme_manager.is_dark_mode()
            bb8_html = bb8_html.replace(
                'type="checkbox"', 
                'type="checkbox" checked' if is_dark else 'type="checkbox"'
            )
            
            # Add JavaScript for theme toggling
            js_code = """
            <script>
            (function() {
                const checkbox = document.querySelector('.bb8-toggle__checkbox');
                if (!checkbox) return;

                checkbox.addEventListener('change', function() {
                    const theme = this.checked ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', theme);
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: { theme: theme }
                    }, '*');
                });
            })();
            </script>
            """
            
            return f"{bb8_html}{js_code}"
        
        # Fallback to basic toggle
        is_dark = theme_manager.is_dark_mode()
        return f"""
        <div class="theme-toggle">
            <label class="toggle-switch">
                <input 
                    type="checkbox" 
                    {"checked" if is_dark else ""}
                    onchange="
                        const theme = this.checked ? 'dark' : 'light';
                        document.documentElement.setAttribute('data-theme', theme);
                        window.parent.postMessage({{
                            type: 'streamlit:setComponentValue',
                            value: {{ theme: theme }}
                        }}, '*');
                    "
                />
                <span class="toggle-slider"></span>
                <span class="toggle-label">{theme_manager.DARK_THEME.capitalize() if is_dark else theme_manager.LIGHT_THEME.capitalize()}</span>
            </label>
        </div>
        """
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the sidebar component.
        
        Returns:
            HTML representation of the sidebar.
        """
        # Get current year and version
        current_year = datetime.now().year
        version = "1.0.0"  # You might want to get this from a config or env variable
        
        # Get navigation buttons and theme toggle HTML
        navigation_html = self.render_navigation_buttons()
        theme_toggle_html = self.render_theme_toggle()
        
        # Try to render using template
        template_content = self.render_template(
            "components/layout/sidebar.html",
            {
                "NAVIGATION_ITEMS": navigation_html,
                "THEME_TOGGLE": theme_toggle_html,
                "VERSION": version,
                "YEAR": current_year
            }
        )
        
        if template_content:
            return template_content
        
        # Fallback to direct HTML generation
        return f"""
        <div class="sidebar-container">
            <div class="sidebar-header">
                <div class="gradient-text">MNIST App</div>
                <div class="sidebar-subheader">Digit Classification AI</div>
            </div>
            
            <div class="nav-buttons-container">
                {navigation_html}
            </div>
            
            <div class="divider"></div>
            
            <div class="toggle-container">
                {theme_toggle_html}
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
        # Add any necessary CSS
        sidebar_css = resource_manager.load_css("components/layout/sidebar.css")
        if sidebar_css:
            resource_manager.inject_css(sidebar_css)
        
        # Create Streamlit sidebar
        with st.sidebar:
            # Inject CSS variables (this ensures sidebar has proper theme styling)
            theme_data = theme_manager.get_theme_data()
            theme_manager._apply_css_variables(theme_data)
            
            # Render sidebar content
            sidebar_html = self.render()
            st.markdown(sidebar_html, unsafe_allow_html=True)
            
            # Handle theme toggle callback
            if "theme" in st.session_state:
                requested_theme = st.session_state["theme"]
                current_theme = theme_manager.get_current_theme()
                
                if requested_theme != current_theme:
                    theme_manager.apply_theme(requested_theme)
                    st.rerun()
            
            # Handle navigation callback
            if "view" in st.session_state:
                from core.app_state.navigation_state import NavigationState
                requested_view = st.session_state["view"]
                current_view = NavigationState.get_active_view()
                
                if requested_view != current_view:
                    NavigationState.set_active_view(requested_view)
                    st.rerun()
