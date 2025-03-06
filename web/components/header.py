import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

def render_header():
    """Render the application header with theme toggle."""
    # Get the header template
    header_html = ResourceLoader.load_template("header.html")
    
    # Get the theme toggle component
    theme_toggle_html = ResourceLoader.load_template(
        "components/theme_toggle.html", 
        THEME_ICON=ThemeManager.get_theme_icon()
    )
    
    # Render the components
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown(theme_toggle_html, unsafe_allow_html=True)
    
    # Create the hidden button for Streamlit state management
    ThemeManager.create_theme_toggle()