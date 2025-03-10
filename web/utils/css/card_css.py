# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/card_css.py
# Description: Utility functions for loading card CSS styles
# Created: 2024-05-01

import streamlit as st
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load_card_css():
    """Load card CSS styles."""
    try:
        # Get the path to the card CSS file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = Path(current_dir).parent.parent
        card_css_path = project_root / "assets" / "css" / "components" / "cards" / "card.css"
        
        # Check if the file exists
        if not card_css_path.exists():
            logger.warning(f"Card CSS file not found at {card_css_path}")
            return
        
        # Read the CSS file
        with open(card_css_path, "r") as css_file:
            css_content = css_file.read()
        
        # Inject the CSS
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.debug("Card CSS loaded successfully")
    
    except Exception as e:
        logger.error(f"Error loading card CSS: {str(e)}", exc_info=True)
        # Provide a fallback minimal card style
        st.markdown("""
        <style>
        .card {
            background-color: var(--color-card, white);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid var(--color-border, #ddd);
        }
        </style>
        """, unsafe_allow_html=True)

def apply_card_styles():
    """Apply card styles to the application."""
    load_card_css() 