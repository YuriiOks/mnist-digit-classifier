# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/debug/font_tester.py
# Description: Font testing utility
# Created: 2024-05-01

import streamlit as st
import logging

# Try to import BaseView, but provide a fallback if it doesn't exist
try:
    from ui.views.base_view import BaseView
    base_view_available = True
except ImportError:
    base_view_available = False
    # Create a simple BaseView replacement
    class BaseView:
        def __init__(self, view_id="font_tester", title="Font Tester", description="", icon="🔤"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon

logger = logging.getLogger(__name__)

class FontTesterView(BaseView):
    """Font tester view for the application."""
    
    def __init__(self):
        """Initialize the font tester view."""
        super().__init__(
            view_id="font_tester",
            title="Font Tester",
            description="Test and debug fonts",
            icon="🔤"
        )
        self.logger = logging.getLogger(f"{__name__}")
    
    def render(self) -> None:
        """Render the font tester view."""
        self.logger.debug("Rendering font tester")
        try:
            st.title("Font Tester")
            
            # Try to force reload fonts
            try:
                from utils.css.font_loader import force_load_fonts
                force_load_fonts()
            except ImportError as e:
                st.warning(f"Could not load font_loader: {str(e)}")
            
            # Add direct font testing elements with inline CSS
            st.markdown("""
            <style>
            .font-test-container {
                padding: 20px; 
                border: 1px solid #ddd; 
                margin-bottom: 20px; 
                border-radius: 5px;
            }
            .font-poppins { font-family: 'Poppins', sans-serif; }
            .font-nunito { font-family: 'Nunito', sans-serif; }
            .font-dm-sans { font-family: 'DM Sans', sans-serif; }
            .font-mono { font-family: 'Roboto Mono', monospace; }
            .font-inter { font-family: 'Inter', sans-serif; }
            </style>
            
            <div class="font-test-container">
                <h1 class="font-poppins">Poppins Heading</h1>
                <p class="font-nunito">This text should be in Nunito font. The quick brown fox jumps over the lazy dog.</p>
                <div class="font-dm-sans">This text should be in DM Sans. The quick brown fox jumps over the lazy dog.</div>
                <code class="font-mono">This should be Roboto Mono. The quick brown fox jumps over the lazy dog.</code>
                <div class="font-inter">This text should be in Inter. The quick brown fox jumps over the lazy dog.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add reloader button
            if st.button("Reload Fonts"):
                try:
                    from utils.css.font_loader import force_load_fonts
                    force_load_fonts()
                    st.success("Fonts reloaded!")
                except ImportError:
                    st.error("Could not load font_loader module")
            
        except Exception as e:
            self.logger.error(f"Error rendering font tester: {str(e)}")
            st.error(f"An error occurred while rendering the font tester: {str(e)}") 