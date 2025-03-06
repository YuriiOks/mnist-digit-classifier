import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

def render_header():
    """Render the application header with theme toggle."""
    # Load JavaScript for theme toggling
    ResourceLoader.load_js(["theme_toggle.js"])
    
    # Get the header template
    header_html = ResourceLoader.load_template("header.html")
    
    # Get the theme toggle component with the appropriate icon
    theme_icon = ThemeManager.get_theme_icon()
    theme_toggle_html = ResourceLoader.load_template(
        "components/theme_toggle.html", 
        THEME_ICON=theme_icon
    )
    
    # Render the components
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown(theme_toggle_html, unsafe_allow_html=True)
