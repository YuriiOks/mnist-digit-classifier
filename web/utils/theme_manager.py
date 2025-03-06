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
            "css/components/theme_toggle.css"
        ])
        
        # Load theme-specific CSS
        if st.session_state.dark_mode:
            ResourceLoader.load_css(["css/themes/dark_mode.css"])
        
        # Load JavaScript for theme toggling
        ResourceLoader.load_js(["theme_toggle.js"])
    
    @staticmethod
    def create_theme_toggle():
        """Create the hidden button that toggles the theme."""
        # Create a hidden button with a specific key for the JavaScript to find
        st.button(
            "🔄", 
            key="dark_mode_toggle", 
            on_click=ThemeManager.toggle_dark_mode,
            type="secondary"
        )
    
    @staticmethod
    def get_theme_icon():
        """Get the appropriate theme icon based on current mode.
        In light mode, show sun (to switch to dark)
        In dark mode, show moon (to switch to light)
        """
        return "☀️" if not st.session_state.dark_mode else "🌙"
    
    @staticmethod
    def get_html_toggle():
        """Get the HTML for the theme toggle button."""
        theme_icon = ThemeManager.get_theme_icon()
        return ResourceLoader.load_template(
            "components/theme_toggle.html", 
            THEME_ICON=theme_icon
        )