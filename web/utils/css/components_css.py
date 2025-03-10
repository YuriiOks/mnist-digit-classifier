# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/components_css.py
# Description: CSS styles for common components
# Created: 2024-05-01

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def apply_card_styles():
    """Apply consistent styling to card components."""
    try:
        from utils.css.card_css import load_card_css
        load_card_css()
    except ImportError:
        logger.warning("Card CSS loader not found, using fallback styles")
        # Fallback card styles (simplified version)
        css = """
        <style>
        /* Card styles that use theme variables */
        .card {
            background-color: var(--color-card, white);
            color: var(--color-text, #212529);
            border-radius: var(--border-radius, 4px);
            border: 1px solid var(--color-border, #dee2e6);
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    logger.debug("Applied card styles")

def apply_button_styles():
    """Apply consistent styling to button components."""
    try:
        from utils.css.button_css import load_button_css
        load_button_css()
    except ImportError:
        logger.warning("Button CSS loader not found, using fallback styles")
        # Fallback button styles
        css = """
        <style>
        .stButton button[kind="primary"] {
            background: linear-gradient(90deg, var(--color-primary, #4F46E5), var(--color-secondary, #06B6D4)) !important;
            color: white !important;
            border: none !important;
            border-radius: 0.5rem !important;
            transition: all 0.3s ease !important;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    logger.debug("Applied button styles")

def apply_component_styles():
    """Apply all component styles."""
    apply_card_styles()
    apply_button_styles()
    
    # Hide automatic view titles
    try:
        from utils.css.title_fix_css import hide_view_titles
        hide_view_titles()
    except ImportError:
        logger.warning("Could not import title fix CSS")
    
    # Add more component style functions as needed 