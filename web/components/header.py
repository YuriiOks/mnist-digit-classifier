import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

def render_header():
    """Render the application header with theme toggle."""
    # Get the header template
    header_html = ResourceLoader.load_template("header.html")
    
    # Render the header
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Get the theme icon from the ThemeManager
    theme_icon = ThemeManager.get_theme_icon()
    
    # Create the theme toggle HTML with the icon directly inserted
    theme_toggle_html = f"""
    <button class="dark-mode-toggle" id="darkModeToggle" title="Toggle dark/light mode">
        <span class="toggle-icon">{theme_icon}</span>
    </button>
    """
    
    st.markdown(theme_toggle_html, unsafe_allow_html=True)
    
    # Create the hidden button that will be clicked by JavaScript
    ThemeManager.create_theme_toggle()