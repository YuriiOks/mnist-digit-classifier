# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/debug/debug_view.py
# Description: Debug view for development and testing
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
        def __init__(self, view_id="debug", title="Debug", description="", icon="🔧"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon

logger = logging.getLogger(__name__)

class DebugView(BaseView):
    """Debug view for development tools."""
    
    def __init__(self):
        """Initialize the debug view."""
        super().__init__(
            view_id="debug",
            title="Debug Tools",
            description="Development and debugging tools",
            icon="🔧"
        )
        self.logger = logging.getLogger(f"{__name__}")
    
    def render(self):
        """Render the debug view."""
        st.title("Debug Tools")
        
        st.write("This is a simple debug view for development purposes.")
        
        # Only show font tools if the imports work
        try:
            from core.app_state.navigation_state import NavigationState
            
            st.subheader("Font Tools")
            
            if st.button("Font Preview"):
                NavigationState.set_active_view("font_preview")
                st.rerun()
                
            if st.button("Force Reload Fonts"):
                from utils.css.font_loader import force_load_fonts
                force_load_fonts()
                st.success("Fonts reloaded!")
                
        except ImportError as e:
            st.error(f"Could not load navigation tools: {str(e)}")

        # Add other utility pages as needed
        st.write("### Other Tools")
        if st.button("🖌️ CSS Debug"):
            from utils.css.css_loader import debug_css
            debug_css() 