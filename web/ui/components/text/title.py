# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/components/text/title.py
# Description: Title component for consistent app headers
# Created: 2024-05-01
# Updated: 2025-03-30

import streamlit as st
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def display_title(
    title: str,
    icon: Optional[str] = None,
    subtitle: Optional[str] = None,
    size: str = "large",
    gradient: bool = False,
):
    """Display a styled title with icon and optional subtitle.

    Args:
        title: The title text
        icon: Optional emoji icon
        subtitle: Optional subtitle text
        size: Size of the title ("small", "medium", "large")
        gradient: Whether to use gradient styling
    """
    logger.debug(f"Displaying title: {title}, size: {size}")

    # Determine classes based on parameters
    title_class = "app-title"
    if size == "large":
        title_class += " text-3xl"
    elif size == "medium":
        title_class += " text-2xl"
    else:
        title_class += " text-xl"

    if gradient:
        title_class += " text-gradient"

    # Create the icon part
    icon_html = f'<span class="icon">{icon}</span>' if icon else ""

    # Create title HTML
    title_html = f"""
    <div class="{title_class}">
        {icon_html}
        <span>{title}</span>
    </div>
    """

    # Add subtitle if provided
    if subtitle:
        subtitle_html = f"""
        <div class="app-subtitle text-muted">
            {subtitle}
        </div>
        """
        title_html = title_html + subtitle_html

    # Display with proper spacing
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
