import streamlit as st
from .resource_loader import ResourceLoader

class ThemeManager:
    """Manages the application theme (light/dark mode)."""
    
    @staticmethod
    def initialize():
        """Initialize theme settings in session state."""
        if 'dark_mode' not in st.session_state:
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
    
    @staticmethod
    def create_theme_toggle():
        """Create the hidden button that toggles the theme."""
        with st.container():
            st.button(
                "🔄", 
                key="dark_mode_toggle", 
                on_click=ThemeManager.toggle_dark_mode
            )
    
    @staticmethod
    def get_theme_icon():
        """Get the appropriate theme icon based on current mode."""
        return " 🌙" if st.session_state.dark_mode else " ☀️" 