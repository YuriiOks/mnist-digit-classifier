# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/sidebar_css.py
# Description: Sidebar CSS loader
# Created: 2024-05-03

import streamlit as st
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load_sidebar_css():
    """Load sidebar CSS styles."""
    try:
        # Get the path to the sidebar CSS file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = Path(current_dir).parent.parent
        sidebar_css_path = project_root / "assets" / "css" / "components" / "layout" / "sidebar.css"
        
        # Check if the file exists
        if not sidebar_css_path.exists():
            logger.warning(f"Sidebar CSS file not found at {sidebar_css_path}")
            return
        
        # Read the CSS file
        with open(sidebar_css_path, "r") as css_file:
            css_content = css_file.read()
        
        # Inject the CSS
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.debug("Sidebar CSS loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading sidebar CSS: {str(e)}", exc_info=True) 