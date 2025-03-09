# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/ui/view_utils.py
# Description: Utilities for view rendering and styling
# Created: 2024-05-01

import streamlit as st
import logging

logger = logging.getLogger(__name__)

def apply_view_styling():
    """Apply consistent styling to all views.
    
    This function injects CSS that ensures all views have the same layout and styling,
    eliminating the need for each view to implement its own layout CSS.
    """
    logger.debug("Applying consistent view styling")
    
    # Apply common view styling
    st.markdown("""
    <style>
    /* Fix content alignment */
    .block-container {
        max-width: 100% !important;
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* View containers */
    .view-container {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Make headers look better */
    h1, h2, h3 {
        margin-bottom: 1rem !important;
        margin-top: 0.5rem !important;
        font-family: var(--font-primary, 'Poppins', sans-serif) !important;
    }
    
    /* Add space around elements */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Remove empty columns */
    .stColumn:empty {
        display: none !important;
    }
    
    /* Ensure cards have consistent styling */
    .card, .content-card {
        margin-bottom: 1.5rem !important;
    }
    
    /* Ensure welcome card is consistent across views */
    .welcome-card {
        margin-bottom: 2rem !important;
    }
    
    /* Two-column layout */
    .two-column-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .two-column-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def create_welcome_card(title, icon, content):
    """Create a consistent welcome card for any view.
    
    Args:
        title: Card title
        icon: Emoji icon to display
        content: HTML content for the card (paragraphs)
        
    Returns:
        str: HTML for the welcome card
    """
    # Format content to ensure paragraphs are properly wrapped
    if not content.startswith('<p>'):
        paragraphs = content.split('\n')
        content = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])
    
    return f"""
    <div class="card card-elevated content-card welcome-card animate-fade-in">
        <div class="card-title">
            <span class="card-icon">{icon}</span>
            {title}
        </div>
        <div class="card-content">
            {content}
        </div>
    </div>
    """
    
def create_section_container(section_id, classes=None):
    """Create a container for a view section with proper styling.
    
    Args:
        section_id: Unique identifier for the section
        classes: Additional CSS classes to apply
        
    Returns:
        str: HTML opening tag for the section container
    """
    class_list = "view-section"
    if classes:
        class_list += f" {classes}"
        
    return f"""<div id="{section_id}" class="{class_list}">"""

def close_section_container():
    """Close a section container.
    
    Returns:
        str: HTML closing tag for the section container
    """
    return "</div>" 