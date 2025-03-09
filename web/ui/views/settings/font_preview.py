# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/settings/font_preview.py
# Description: Font preview page to see available fonts
# Created: 2024-05-01

import streamlit as st
import logging

from ui.views.base_view import BaseView
from ui.components.cards.card import Card

logger = logging.getLogger(__name__)

class FontPreviewView(BaseView):
    """Font preview view for the application."""
    
    def __init__(self):
        """Initialize the font preview view."""
        super().__init__(
            view_id="font_preview",
            title="Font Preview",
            description="Preview the available fonts",
            icon="🔤"
        )
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
    
    def render(self) -> None:
        """Render the font preview."""
        self.logger.debug("Rendering font preview")
        try:
            st.title("Font Preview")
            st.markdown("This page shows the available fonts for the application.")
            
            # Display primary font
            st.markdown("## Primary Font (Poppins)")
            st.markdown('<div class="text-primary">', unsafe_allow_html=True)
            st.markdown("**Regular text:** The quick brown fox jumps over the lazy dog.")
            st.markdown("**Bold text:** <strong>The quick brown fox jumps over the lazy dog.</strong>", unsafe_allow_html=True)
            st.markdown("**Semibold:** <span style='font-weight: 600'>The quick brown fox jumps over the lazy dog.</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display secondary font
            st.markdown("## Secondary Font (Nunito)")
            st.markdown('<div class="text-secondary">', unsafe_allow_html=True)
            st.markdown("**Regular text:** The quick brown fox jumps over the lazy dog.")
            st.markdown("**Bold text:** <strong>The quick brown fox jumps over the lazy dog.</strong>", unsafe_allow_html=True)
            st.markdown("**Semibold:** <span style='font-weight: 600'>The quick brown fox jumps over the lazy dog.</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display UI font
            st.markdown("## UI Font (DM Sans)")
            st.markdown('<div class="text-ui">', unsafe_allow_html=True)
            st.markdown("**Regular text:** The quick brown fox jumps over the lazy dog.")
            st.markdown("**Bold text:** <strong>The quick brown fox jumps over the lazy dog.</strong>", unsafe_allow_html=True)
            st.markdown("**Semibold:** <span style='font-weight: 600'>The quick brown fox jumps over the lazy dog.</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display mono font
            st.markdown("## Mono Font (Roboto Mono)")
            st.markdown('<div class="text-mono">', unsafe_allow_html=True)
            st.markdown("**Regular text:** The quick brown fox jumps over the lazy dog.")
            st.markdown("**Bold text:** <strong>The quick brown fox jumps over the lazy dog.</strong>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display digit preview
            st.markdown("## Digit Display (Mono Font)")
            st.markdown('<div class="digit-display" style="font-size: 3rem; line-height: 1; text-align: center;">', unsafe_allow_html=True)
            st.markdown("0 1 2 3 4 5 6 7 8 9", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display heading styles
            st.markdown("## Heading Styles")
            st.markdown('<h1 class="heading-hero">Hero Heading</h1>', unsafe_allow_html=True)
            st.markdown('<h2 class="heading-section">Section Heading</h2>', unsafe_allow_html=True)
            
            # Special text styles
            st.markdown("## Special Text Styles")
            st.markdown('<div class="text-gradient" style="font-size: 2rem; font-weight: bold;">Gradient Text Style</div>', unsafe_allow_html=True)
            st.markdown('<div class="badge">Badge Style</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## Font Weights")
            weights = ["light (300)", "regular (400)", "medium (500)", "semibold (600)", "bold (700)"]
            for weight in weights:
                weight_value = weight.split(" ")[0]
                weight_num = weight.split("(")[1].split(")")[0]
                st.markdown(f'<div style="font-weight: {weight_num};" class="text-primary">{weight_value}: The quick brown fox jumps over the lazy dog.</div>', unsafe_allow_html=True)
                
            self.logger.debug("Font preview rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering font preview: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering the font preview: {str(e)}") 