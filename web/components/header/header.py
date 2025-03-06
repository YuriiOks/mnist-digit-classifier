import streamlit as st
import os
from pathlib import Path
from .theme_toggle import render_theme_toggle

def render_header():
    """Render the application header with logo and title."""
    
    # Determine component directory
    component_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(component_dir)))
    
    # Load CSS for header
    css_path = os.path.join(app_dir, "web", "static", "css", "components", "header.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Load header template
    template_path = os.path.join(app_dir, "web", "templates", "header.html")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            header_template = f.read()
        
        # Render the header template
        st.markdown(header_template, unsafe_allow_html=True)
    else:
        # Fallback to inline HTML if template is not found
        st.markdown("""
        <div class="header-container">
            <div class="header-logo-title">
                <div class="header-logo">✏️</div>
                <h1 class="header-title">MNIST Digit Classifier</h1>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Render the theme toggle component
    render_theme_toggle() 