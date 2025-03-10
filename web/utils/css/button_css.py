# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/button_css.py
# Description: Utility functions for loading button CSS styles
# Created: 2024-05-01

import streamlit as st
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load_button_css():
    """Load button CSS styles."""
    try:
        # Get the path to the button CSS file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = Path(current_dir).parent.parent
        button_css_path = project_root / "assets" / "css" / "components" / "buttons" / "button.css"
        
        # Check if the file exists
        if not button_css_path.exists():
            logger.warning(f"Button CSS file not found at {button_css_path}")
            return
        
        # Read the CSS file
        with open(button_css_path, "r") as css_file:
            css_content = css_file.read()
        
        # Inject the CSS
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.debug("Button CSS loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading button CSS: {str(e)}", exc_info=True)
        # Provide a fallback minimal button style
        st.markdown("""
        <style>
        .stButton button[kind="primary"] {
            background-color: #4F46E5 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

def apply_button_styles():
    """Apply button styles to the application."""
    load_button_css() 