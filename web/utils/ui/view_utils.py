# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/ui/view_utils.py
# Description: Utilities for view rendering and styling
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

def apply_view_styling():
    """Apply consistent styling to all views by loading external CSS."""
    logger.info("Applying consistent view styling")
    
    # Get the path to the CSS file
    current_dir = Path(__file__).resolve().parent
    css_path = current_dir.parent.parent / "assets" / "css" / "views" / "view_styles.css"
    
    logger.info(f"View styles CSS path: {css_path}")

    if css_path.exists():
        # Load and inject the CSS
        with open(css_path, "r") as css_file:
            css_content = css_file.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        logger.debug("View styling applied successfully")
    else:
        logger.warning(f"View styles CSS file not found at {css_path}")

def create_card(title: str, icon: str, content: str,
                template_loader: Callable[[str], str],
                type_card: str = "welcome") -> str:
    """Create a consistent welcome card for any view.
    
    Args:
        title: Card title
        icon: Emoji icon to display
        content: HTML content for the card (paragraphs)
        
    Returns:
        str: HTML for the welcome card
    """
    
    formatted_content = ''.join(f'<p>{p.strip()}</p>' for p in content.split('\n') if p.strip())
    return template_loader(f"/components/controls/cards/{type_card}_card.html", {
        "title": title,
        "icon": icon,
        "content": formatted_content
    })
    
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