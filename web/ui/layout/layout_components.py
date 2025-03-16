# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/layout_components.py
# Description: Layout components for the application
# Created: 2025-03-16

import streamlit as st
import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from ui.components.base.component import Component
from ui.theme.theme_manager import theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils

class Header(Component[None]):
    """Header component for the application."""
    
    def __init__(
        self,
        title: str = "MNIST Digit Classifier",
        actions_html: str = "",
        toggle_theme_callback: Optional[Callable] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the header component.
        
        Args:
            title: Application title to display.
            actions_html: HTML for additional actions in the header.
            toggle_theme_callback: Function to call when theme toggle is clicked.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
            key: Unique key for Streamlit.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_type="layout",
            component_name="header",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )
        self.__title = title
        self.__actions_html = actions_html
        self.__toggle_theme_callback = toggle_theme_callback or theme_manager.toggle_theme
    
    @property
    def title(self) -> str:
        """Get the header title."""
        return self.__title
    
    @title.setter
    def title(self, value: str) -> None:
        """Set the header title."""
        self.__title = value
    
    @property
    def actions_html(self) -> str:
        """Get the header actions HTML."""
        return self.__actions_html
    
    @actions_html.setter
    def actions_html(self, value: str) -> None:
        """Set the header actions HTML."""
        self.__actions_html = value
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the header component.
        
        Returns:
            HTML representation of the header.
        """
        # Try to render using template
        template_content = self.render_template(
            "components/layout/header.html",
            {
                "TITLE": self.__title,
                "ACTIONS_HTML": self.__actions_html
            }
        )
        
        if template_content:
            return template_content
        
        # Fallback to direct HTML generation
        return f"""
        <div class="app-header">
            <h1>{self.__title}</h1>
            {self.__actions_html}
        </div>
        """
    
    @AspectUtils.catch_errors
    def display(self) -> None:
        """Display the header component in Streamlit."""
        # Render the HTML
        header_html = self.render()
        
        # Display in Streamlit
        st.markdown(header_html, unsafe_allow_html=True)


class Footer(Component[None]):
    """Footer component for the application."""
    
    def __init__(
        self,
        content: Optional[str] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the footer component.
        
        Args:
            content: Footer content (HTML).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dict of HTML attributes to apply to the component.
            key: Unique key for Streamlit.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            component_type="layout",
            component_name="footer",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )
        
        # Set default content if none provided
        self.__content = content or self._get_default_content()
    
    @property
    def content(self) -> str:
        """Get the footer content."""
        return self.__content
    
    @content.setter
    def content(self, value: str) -> None:
        """Set the footer content."""
        self.__content = value
    
    def _get_default_content(self) -> str:
        """
        Get default footer content.
        
        Returns:
            Default footer content with current year.
        """
        current_year = datetime.now().year
        return f"""
        © {current_year} MNIST Digit Classifier | 
        Developed by <a href="https://github.com/YuriODev" target="_blank">YuriODev</a> | 
        <a href="https://github.com/YuriiOks/mnist-digit-classifier" target="_blank">
        <span style="white-space: nowrap;">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" 
        viewBox="0 0 16 16" style="vertical-align: text-bottom; margin-right: 4px;">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 
        0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-
        .28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-
        .87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-
        2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 
        2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-
        3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 
        8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
        </svg>GitHub</span></a>
        """
    
    @AspectUtils.catch_errors
    def render(self) -> str:
        """
        Render the footer component.
        
        Returns:
            HTML representation of the footer.
        """
        # Try to render using template
        template_content = self.render_template(
            "components/layout/footer.html",
            {
                "CONTENT": self.__content,
                "YEAR": datetime.now().year
            }
        )
        
        if template_content:
            return template_content
        
        # Fallback to direct HTML generation
        return f"""
        <footer class="app-footer">
            <div class="footer-content">
                {self.__content}
            </div>
        </footer>
        """
    
    @AspectUtils.catch_errors
    def display(self) -> None:
        """Display the footer component in Streamlit."""
        # Render the HTML
        footer_html = self.render()
        
        # Display in Streamlit
        st.markdown(footer_html, unsafe_allow_html=True)


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
        # Display sidebar
        self.sidebar.display()
        
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