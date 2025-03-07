import streamlit as st
import json
import os
from streamlit_option_menu import option_menu
from utils.theme_manager import ThemeManager
from utils.resource_loader import ResourceLoader

def render_sidebar():
    """Render the sidebar with navigation and theme toggle."""
    with st.sidebar:
        # Load sidebar CSS
        ResourceLoader.load_css([
            "css/components/sidebar.css",
            "css/components/navigation.css"
        ])
        
        # Render theme toggle
        render_theme_toggle()
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # Render navigation menu
        selected = render_navigation()
        
        return selected

def load_navigation_config():
    """Load navigation configuration from JSON file."""
    try:
        config_path = os.path.join(
            ResourceLoader.get_app_dir(),
            "static",
            "config",
            "navigation.json"
        )
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading navigation config: {str(e)}")
        # Fallback default configuration
        return {
            "menu": {
                "title": "",
                "items": [
                    {"name": "Home", "icon": "house"},
                    {"name": "Drawing", "icon": "pencil"},
                    {"name": "History", "icon": "clock-history"},
                    {"name": "Settings", "icon": "gear"}
                ],
                "icon": "cast",
                "defaultIndex": 0
            }
        }

def render_navigation():
    """Render the navigation menu based on configuration."""
    # Load navigation configuration
    config = load_navigation_config()
    menu_config = config["menu"]
    
    # Get navigation items
    items = [item["name"] for item in menu_config["items"]]
    icons = [item["icon"] for item in menu_config["items"]]
    
    # Add mode-specific class for styling
    mode_class = "dark-mode" if st.session_state.dark_mode else "light-mode"
    st.markdown(f'<div class="{mode_class}" id="navigation-wrapper">', unsafe_allow_html=True)
    
    # Render option menu
    selected = option_menu(
        menu_config["title"], 
        items, 
        icons=icons, 
        menu_icon=menu_config["icon"],
        default_index=menu_config["defaultIndex"],
        styles={
            "container": {"padding": "5px", "background-color": "transparent"},
            "icon": {"color": "#4CA1AF", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "#eee",
                "transition": "all 0.3s ease",
            },
            "nav-link-selected": {
                "background-color": "#2C3E50",
                "font-weight": "600",
            },
            "menu-title": {
                "margin-bottom": "15px",
                "font-size": "18px",
                "color": "#2C3E50" if not st.session_state.dark_mode else "#64c4d2"
            }
        }
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return selected

def render_theme_toggle():
    """Render the theme toggle component in the sidebar."""
    # Get the theme icon from the ThemeManager
    theme_icon = ThemeManager.get_theme_icon()
    
    # Load the theme toggle template
    sidebar_toggle_html = ResourceLoader.load_template(
        "components/sidebar_toggle.html",
        THEME_ICON=theme_icon
    )
    
    # Render the theme toggle
    st.markdown(sidebar_toggle_html, unsafe_allow_html=True) 