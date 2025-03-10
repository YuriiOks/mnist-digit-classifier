import streamlit as st
from typing import Callable, Optional
from ui.theme.theme_manager import ThemeManager

class ThemeToggle:
    """A toggle component for switching between light and dark themes."""
    
    def __init__(self, theme_manager: ThemeManager, on_change: Optional[Callable] = None):
        self.theme_manager = theme_manager
        self.on_change = on_change
        
        # Initialize session state for theme if not already present
        if "current_theme" not in st.session_state:
            st.session_state.current_theme = "light"
    
    def render(self):
        """Render the theme toggle in the Streamlit UI."""
        # Get available themes
        themes = self.theme_manager.get_available_themes()
        
        # Create a selectbox for theme switching
        selected_theme = st.selectbox(
            "Theme",
            options=list(themes.keys()),
            format_func=lambda x: themes[x],
            index=list(themes.keys()).index(st.session_state.current_theme) 
                if st.session_state.current_theme in themes else 0,
            key="theme_selector"
        )
        
        # Apply the selected theme
        if selected_theme != st.session_state.current_theme:
            st.session_state.current_theme = selected_theme
            self.theme_manager.apply_theme(selected_theme)
            
            # Call the on_change callback if provided
            if self.on_change:
                self.on_change(selected_theme) 