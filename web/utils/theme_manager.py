import streamlit as st
from .resource_loader import ResourceLoader

class ThemeManager:
    """Manages the application theme (light/dark mode)."""
    
    @staticmethod
    def initialize():
        """Initialize theme settings in session state."""
        if 'dark_mode' not in st.session_state:
            # Default to light mode (sun icon showing)
            st.session_state.dark_mode = False
        
        # Check if theme toggle was triggered via query parameter
        if 'toggle_theme' in st.query_params:
            # Remove the parameter to prevent repeated toggling
            st.query_params.clear()
            # Toggle the theme
            ThemeManager.toggle_dark_mode()
    
    @staticmethod
    def toggle_dark_mode():
        """Toggle between light and dark mode."""
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    @staticmethod
    def load_theme_resources():
        """Load the appropriate CSS based on current theme."""
        # Load base CSS that's common to all themes
        ResourceLoader.load_css(["css/base.css"])
        ResourceLoader.load_css([
            "css/components/header.css", 
            "css/components/footer.css", 
            "css/components/theme_toggle.css",
            "css/components/sidebar.css"
        ])
        
        # Load theme-specific CSS
        if st.session_state.dark_mode:
            ResourceLoader.load_css(["css/themes/dark_mode.css"])
        
        # Load JavaScript for theme toggling
        ResourceLoader.load_js(["theme_toggle.js"])
    
    @staticmethod
    def get_theme_icon():
        """Get the appropriate theme icon based on current mode.
        In light mode, show moon icon (to switch to dark)
        In dark mode, show sun icon (to switch to light)
        """
        # Just return the icon character, NOT a Streamlit button
        return "☀️" if not st.session_state.dark_mode else "🌙"