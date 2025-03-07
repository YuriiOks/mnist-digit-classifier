import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

def render_header():
    """Render the application header with theme toggle."""
    # Get the header template
    header_html = ResourceLoader.load_template("header.html")
    
    # Render the header
    st.markdown(header_html, unsafe_allow_html=True)
    
    # The floating theme toggle button has been removed
    # Theme toggle is now only available in the sidebar