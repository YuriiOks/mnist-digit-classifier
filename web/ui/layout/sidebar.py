# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/sidebar.py
# Description: Sidebar component for the application
# Created: 2025-03-17

import streamlit as st
import logging
import traceback
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
        # Create a logger for debugging before initializing parent
        self._init_logger = logging.getLogger(f"{__name__}.Sidebar.__init__")
        self._init_logger.info("Initializing Sidebar component")
        
        try:
            super().__init__(
                component_type="layout",
                component_name="sidebar",
                id=id,
                classes=classes or [],
                attributes=attributes or {},
                key=key or "app_sidebar",
                **kwargs
            )
            self._init_logger.info("Sidebar component parent initialized successfully")
        except Exception as e:
            self._init_logger.error(f"Error initializing Sidebar parent: {str(e)}", exc_info=True)
            raise
    
    @AspectUtils.catch_errors
    def render_navigation_buttons(self) -> str:
        """
        Render the navigation buttons.
        
        Returns:
            HTML for navigation buttons.
        """
        self._logger.info("Rendering navigation buttons")
        
        try:
            # Navigation items
            from core.app_state.navigation_state import NavigationState
            nav_items = NavigationState.get_routes()
            
            # Get active view
            active_view = NavigationState.get_active_view()
            
            buttons_html = ""
            for item in nav_items:
                view_id = item["id"]
                label = f"{item['icon']} {item['label']}"
                is_active = view_id == active_view
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    label,
                    key=f"nav_{view_id}",  # Note: just use nav_home, not nav_home_btn
                    use_container_width=True,
                    type=button_type
                ):
                    # Your navigation logic
                    NavigationState.set_active_view(view_id)
                    st.rerun()

                active_view = NavigationState.get_active_view()
    
            
            self._logger.info(f"Generated navigation buttons HTML for {len(nav_items)} items")
            return buttons_html
        except Exception as e:
            self._logger.error(f"Error rendering navigation buttons: {str(e)}", exc_info=True)
            return "<div>Error loading navigation</div>"
    
    @AspectUtils.catch_errors
    def render_theme_toggle(self) -> str:
        """
        Render the theme toggle.
        
        Returns:
            HTML for theme toggle component.
        """
        self._logger.info("Rendering theme toggle")
        
        try:
            # Try to load BB8 toggle component
            bb8_html = resource_manager.load_template("components/controls/bb8-toggle.html")
            if bb8_html:
                self._logger.info("Found BB8 toggle template")
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
            else:
                self._logger.warning("BB8 toggle template not found, using fallback")
            
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
        except Exception as e:
            self._logger.error(f"Error rendering theme toggle: {str(e)}", exc_info=True)
            return "<div>Error loading theme toggle</div>"
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the sidebar component.
        
        Returns:
            HTML representation of the sidebar.
        """
        self._logger.info("Rendering sidebar HTML")
        
        try:
            # Get current year and version
            current_year = datetime.now().year
            version = "1.0.0"  # You might want to get this from a config or env variable
            
            # Get navigation buttons and theme toggle HTML
            navigation_html = self.render_navigation_buttons()
            theme_toggle_html = self.render_theme_toggle()
            
            # Try to render using template
            self._logger.info("Attempting to render sidebar using template")
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
                self._logger.info("Successfully rendered sidebar using template")
                return template_content
            
            # Fallback to direct HTML generation
            self._logger.info("Falling back to direct HTML generation for sidebar")
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
        except Exception as e:
            self._logger.error(f"Error rendering sidebar: {str(e)}", exc_info=True)
            return "<div>Error rendering sidebar</div>"
    
    @AspectUtils.catch_errors
    def display(self) -> None:
        """Display the sidebar component in Streamlit."""
        self._logger.info("Displaying sidebar")
        
        try:
            # Load CSS for sidebar
            self._logger.info("Loading sidebar CSS")
            sidebar_css = resource_manager.load_css("components/layout/sidebar.css")
            if sidebar_css:
                resource_manager.inject_css(sidebar_css)
                self._logger.info("Loaded sidebar CSS")
            else:
                self._logger.warning("Failed to load sidebar CSS")
            
            # IMPORTANT: Use Streamlit's built-in sidebar container
            self._logger.info("Using Streamlit's sidebar container")
            with st.sidebar:
                try:
                    # Inject CSS variables for proper theme styling
                    self._logger.info("Applying theme CSS variables")
                    theme_data = theme_manager.get_theme_data()
                    theme_manager._apply_css_variables(theme_data)
                    
                    # Instead of rendered HTML, use direct Streamlit widgets
                    self._logger.info("Using direct Streamlit widgets for sidebar")
                    
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
                    from core.app_state.navigation_state import NavigationState
                    nav_items = NavigationState.get_routes()
                    active_view = NavigationState.get_active_view()
                    
                    # Spacer
                    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
                    
                    # Create navigation buttons
                    self._logger.info("Creating navigation buttons")
                    for item in nav_items:
                        if st.button(
                            f"{item['icon']} {item['label']}",
                            key=f"nav_{item['id']}",
                            type="primary" if active_view == item['id'] else "secondary",
                            use_container_width=True
                        ):
                            self._logger.info(f"Navigation button clicked: {item['id']}")
                            NavigationState.set_active_view(item['id'])
                            st.rerun()
                    script = f"""
                            <script>
                                // Add this script to run when the DOM is ready
                                (function() {{
                                    // Wait a moment for Streamlit to fully render
                                    setTimeout(function() {{
                                        // Find active navigation button and add the class
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
                    
                    # BB8 Toggle Container
                    self._logger.info("Creating BB8 toggle")
                    with st.container():
                        # Instantiate BB8Toggle
                        bb8_toggle = BB8Toggle(
                            theme_manager_instance=theme_manager,
                            on_change=lambda new_theme: st.rerun(),  # reload to apply new theme
                            key="sidebar_bb8_toggle"
                        )
                        # Display the toggle; it handles theme switching internally
                        bb8_toggle.display()
                    
                    # Another divider
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    # Footer with version/year
                    current_year = datetime.now().year
                    version = "1.0.0"  # Or load from config
                    
                    st.markdown(
                        f"""
                        <div class="sidebar-footer">
                            <p>Version {version}</p>
                            <p>© {current_year} MNIST Classifier</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    self._logger.info("Direct Streamlit sidebar widgets created successfully")
                    
                except Exception as e:
                    self._logger.error(f"Error in Streamlit sidebar container: {str(e)}", exc_info=True)
                    st.error("Error rendering sidebar components")
                
                # Handle theme toggle callback
                if "theme" in st.session_state:
                    requested_theme = st.session_state["theme"]
                    current_theme = theme_manager.get_current_theme()
                    
                    if requested_theme != current_theme:
                        self._logger.info(f"Theme change detected: {current_theme} -> {requested_theme}")
                        theme_manager.apply_theme(requested_theme)
                        st.rerun()
                
                # Handle navigation callback
                if "view" in st.session_state:
                    from core.app_state.navigation_state import NavigationState
                    requested_view = st.session_state["view"]
                    current_view = NavigationState.get_active_view()
                    
                    if requested_view != current_view:
                        self._logger.info(f"View change detected: {current_view} -> {requested_view}")
                        NavigationState.set_active_view(requested_view)
                        st.rerun()
        except Exception as e:
            self._logger.error(f"Error displaying sidebar: {str(e)}")
            self._logger.error(traceback.format_exc())
            st.sidebar.error("Error rendering application sidebar")