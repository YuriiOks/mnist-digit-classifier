# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/ui/layout.py
# Description: Basic layout utilities for fixing UI issues
# Created: 2024-05-01

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def fix_layout_issues():
    """Apply minimal fixes for layout issues with header and footer."""
    st.markdown("""
    <style>
    /* Basic fixes for header and footer alignment */
    .app-header, .app-footer {
        text-align: center !important;
    }
    
    /* Ensure content takes full width */
    .block-container {
        max-width: 100% !important;
        padding: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    logger.debug("Applied minimal layout fixes") 