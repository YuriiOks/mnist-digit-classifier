import streamlit as st
from .resource_loader import ResourceLoader

class ThemeManager:
    """Manages the application theme (light/dark mode)."""
    
    @staticmethod
    def initialize():
        """Initialize theme settings in session state."""
        # Initialize theme config if not already in session state
        if 'theme_config' not in st.session_state:
            st.session_state.theme_config = {
                "current_theme": "light",
                "refreshed": True,
                
                "light": {
                    "theme.base": "light",
                    "theme.backgroundColor": "white",
                    "theme.primaryColor": "#4CA1AF",
                    "theme.secondaryBackgroundColor": "#f8f9fa",
                    "theme.textColor": "#333",
                    "icon": "☀️"  # Moon icon to switch to dark mode
                },
                
                "dark": {
                    "theme.base": "dark",
                    "theme.backgroundColor": "#121212",
                    "theme.primaryColor": "#64c4d2",
                    "theme.secondaryBackgroundColor": "#1e1e1e",
                    "theme.textColor": "#f0f0f0",
                    "icon": "🌙"  # Sun icon to switch to light mode
                }
            }
            
            # Set initial dark_mode flag for backwards compatibility
            st.session_state.dark_mode = (st.session_state.theme_config["current_theme"] == "dark")
        
        # Check if theme toggle was triggered via query parameter
        if 'toggle_theme' in st.query_params:
            # Remove the parameter to prevent repeated toggling
            st.query_params.clear()
            # Toggle the theme
            ThemeManager.toggle_dark_mode()
    
    @staticmethod
    def toggle_dark_mode():
        """Toggle between light and dark mode using direct Streamlit config."""
        # Store the previous theme
        previous_theme = st.session_state.theme_config["current_theme"]
        
        # Get the theme dictionary for the current theme
        theme_dict = st.session_state.theme_config["dark"] if previous_theme == "light" else st.session_state.theme_config["light"]
        
        # Set each theme property directly in Streamlit config
        for key, value in theme_dict.items():
            if key.startswith("theme"):
                st._config.set_option(key, value)
        
        # Update the current theme
        new_theme = "dark" if previous_theme == "light" else "light"
        st.session_state.theme_config["current_theme"] = new_theme
        
        # Update dark_mode flag for backwards compatibility
        st.session_state.dark_mode = (new_theme == "dark")
        
        # Mark as needing refresh
        st.session_state.theme_config["refreshed"] = False
    
    @staticmethod
    def load_theme_resources():
        """Load the appropriate CSS based on current theme."""
        # Check if we need to refresh after a theme change
        if st.session_state.theme_config.get("refreshed") == False:
            st.session_state.theme_config["refreshed"] = True
            st.rerun()
        
        # Load base CSS that's common to all themes
        ResourceLoader.load_css(["css/base.css"])
        ResourceLoader.load_css([
            "css/components/header.css", 
            "css/components/footer.css", 
            "css/components/theme_toggle.css",
            "css/components/sidebar.css",
            "css/components/navigation.css"
        ])
        
        # Load theme-specific CSS for additional customizations
        if st.session_state.dark_mode:
            ResourceLoader.load_css(["css/themes/dark_mode.css"])
        
        # Load JavaScript for theme toggling
        ResourceLoader.load_js(["theme_toggle.js"])
    
    @staticmethod
    def get_theme_icon():
        """Get the appropriate theme icon based on current mode."""
        current_theme = st.session_state.theme_config["current_theme"]
        return st.session_state.theme_config[current_theme]["icon"]