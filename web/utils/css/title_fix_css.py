# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/title_fix_css.py
# Description: CSS fixes for hiding automatic view titles and descriptions
# Created: 2024-05-01

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def hide_view_titles():
    """Apply CSS to hide view titles and descriptions."""
    css = """
    <style>
    /* Hide the default view titles */
    h1.view-title {
        display: none !important;
    }
    
    /* Hide view descriptions */
    p.view-description {
        display: none !important;
    }
    
    /* Target the container that holds view descriptions */
    .stElementContainer:has(p.view-description) {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ensure there's no empty space where the title would be */
    .stMarkdown:empty {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    logger.debug("Applied view title and description hiding CSS") 