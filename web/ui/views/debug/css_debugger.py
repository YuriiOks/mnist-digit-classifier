# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/debug/css_debugger.py
# Description: CSS debugging tool
# Created: 2024-05-01

import streamlit as st
import logging
import os
from pathlib import Path

# Try to import BaseView, but provide a fallback if it doesn't exist
try:
    from ui.views.base_view import BaseView
    base_view_available = True
except ImportError:
    base_view_available = False
    # Create a simple BaseView replacement
    class BaseView:
        def __init__(self, view_id="css_debug", title="CSS Debugger", description="", icon="🎨"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon

logger = logging.getLogger(__name__)

class CSSDebugView(BaseView):
    """CSS debug view for the application."""
    
    def __init__(self):
        """Initialize the CSS debug view."""
        super().__init__(
            view_id="css_debug",
            title="CSS Debugger",
            description="Debug CSS issues",
            icon="🎨"
        )
        self.logger = logging.getLogger(f"{__name__}")
    
    def render(self) -> None:
        """Render the CSS debug view."""
        self.logger.debug("Rendering CSS debug view")
        try:
            st.title("CSS Debugger")
            
            # Section for font testing
            st.header("Font Testing")
            
            # Try to force reload fonts
            try:
                from utils.css.font_loader import force_load_fonts
                force_load_fonts()
            except ImportError as e:
                st.warning(f"Could not load font_loader: {str(e)}")
            
            # Add direct font testing 
            st.markdown("""
            <div style="border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px;">
                <h3 style="font-family: 'Poppins', sans-serif;">This heading should be in Poppins</h3>
                <p style="font-family: 'Nunito', sans-serif;">This paragraph should be in Nunito. The quick brown fox jumps over the lazy dog.</p>
                <pre style="font-family: 'Roboto Mono', monospace;">This is a code block in Roboto Mono</pre>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a button to test font loading
            if st.button("Force reload fonts"):
                try:
                    from utils.css.font_loader import force_load_fonts
                    force_load_fonts()
                    st.success("Fonts reloaded!")
                except ImportError as e:
                    st.error(f"Could not load font_loader: {str(e)}")
            
            # Section for checking CSS files
            st.header("CSS File Check")
            
            css_files = [
                "assets/css/global/fonts.css",
                "assets/css/global/variables.css",
                "assets/css/global/typography.css",
                "assets/css/global/cards.css",
                "assets/css/components/cards/card.css",
                "assets/css/components/cards/content_card.css",
            ]
            
            st.write("Checking if CSS files exist:")
            for css_path in css_files:
                full_path = os.path.join(os.getcwd(), css_path)
                file_exists = os.path.exists(full_path)
                status = "✅ Exists" if file_exists else "❌ Missing"
                st.write(f"- {css_path}: {status}")
                
                if file_exists and st.checkbox(f"View {css_path}", key=f"view_{css_path}"):
                    with open(full_path, 'r') as f:
                        content = f.read()
                        st.code(content, language="css")
            
            # Test applying CSS directly
            st.header("Direct CSS Injection Test")
            
            test_css = """
            <style>
            .test-box {
                background-color: #4338CA;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-family: 'Nunito', sans-serif;
                margin-top: 20px;
            }
            
            .test-box h4 {
                font-family: 'Poppins', sans-serif;
                margin-top: 0;
            }
            
            .test-box code {
                font-family: 'Roboto Mono', monospace;
                background-color: rgba(255,255,255,0.2);
                padding: 2px 5px;
                border-radius: 3px;
            }
            </style>
            
            <div class="test-box">
                <h4>CSS Test Box</h4>
                <p>If this box is blue with white text, direct CSS injection works.</p>
                <p>This text should be in <code>Nunito</code>, the heading in <code>Poppins</code>, and this code in <code>Roboto Mono</code>.</p>
            </div>
            """
            
            st.markdown(test_css, unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Error rendering CSS debug view: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}") 