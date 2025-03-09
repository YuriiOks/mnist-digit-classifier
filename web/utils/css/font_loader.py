# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/font_loader.py
# Description: Font loading utility
# Created: 2024-05-01

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def force_load_fonts():
    """Force load Google Fonts and apply them directly to elements."""
    logger.info("Forcing font loading")
    
    # Direct injection of font CSS - Using !important to override Streamlit styles
    font_css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap');
    
    /* Define variables */
    :root {
        --font-primary: 'Poppins', sans-serif;
        --font-secondary: 'Nunito', sans-serif;
        --font-mono: 'Roboto Mono', monospace;
    }
    
    /* Apply fonts to elements with high specificity to override Streamlit */
    body, .stApp, div.stApp > div {
        font-family: var(--font-secondary) !important;
    }
    
    h1, h2, h3, h4, h5, h6, 
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    div[data-testid="stHeader"] {
        font-family: var(--font-primary) !important;
    }
    
    code, pre, .stCodeBlock, 
    div[data-testid="stCodeBlock"] pre {
        font-family: var(--font-mono) !important;
    }
    
    /* Apply to Streamlit's markdown containers */
    .stMarkdown, .stText, 
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] div {
        font-family: var(--font-secondary) !important;
    }
    
    /* High specificity for buttons */
    button, .stButton button {
        font-family: var(--font-primary) !important;
    }
    
    /* Custom components */
    .card, .content-card {
        font-family: var(--font-secondary) !important;
    }
    
    .card-title, .card .card-title, .content-card .card-title {
        font-family: var(--font-primary) !important;
    }
    </style>
    """
    
    # Inject the font CSS 
    st.markdown(font_css, unsafe_allow_html=True)
    
    # Also add a script to apply fonts after the page loads
    script = """
    <script>
    // Wait for the document to be fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        // Apply fonts to body
        document.body.style.fontFamily = "var(--font-secondary)";
        
        // Apply fonts to headings
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headings.forEach(h => {
            h.style.fontFamily = "var(--font-primary)";
        });
        
        console.log("Fonts applied via JavaScript");
    });
    </script>
    """
    st.markdown(script, unsafe_allow_html=True)
    
    logger.info("Font loading complete")

def load_app_fonts():
    """Load fonts for the app at startup."""
    logger.info("Loading application fonts")
    
    # Create a direct link to Google Fonts
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Nunito:wght@300;400;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
        """, 
        unsafe_allow_html=True
    )
    
    # Apply fonts directly
    force_load_fonts()
    
    logger.info("Application fonts loaded") 