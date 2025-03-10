# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/theme/theme_manager.py
# Description: Theme management for the application
# Created: 2024-05-01

import streamlit as st
import json
import os
from typing import Dict, Any, Literal, Optional
import logging
from pathlib import Path

from core.app_state.theme_state import ThemeState
from utils.file.path_utils import get_project_root
from utils.css.css_loader import load_css, load_theme_css, inject_css, load_css_file
from utils.js.js_loader import load_js

logger = logging.getLogger(__name__)

ThemeType = Literal["light", "dark"]

class ThemeManager:
    """Manages theme loading, switching, and application throughout the app."""
    
    def __init__(self):
        self.themes = {}
        self.current_theme = None
        self.themes_dir = Path("assets/config/themes")
        self._load_themes()
    
    def _load_themes(self) -> None:
        """Load all theme files from the themes directory."""
        for theme_file in self.themes_dir.glob("*.json"):
            if theme_file.name == "default.json":
                continue  # Skip default as it's loaded as a base
                
            with open(theme_file, "r") as f:
                theme_data = json.load(f)
                theme_name = theme_data.get("name", theme_file.stem)
                self.themes[theme_name] = theme_data
    
    def get_theme(self, name: str) -> Dict[str, Any]:
        """Get a theme by name, with fallback to default."""
        # Always load the default theme as the base
        with open(self.themes_dir / "default.json", "r") as f:
            theme = json.load(f)
            
        # If requesting a specific theme, overlay it on default
        if name != "default" and name in self.themes:
            # Merge with priority to the requested theme for any overlapping keys
            self._deep_merge(theme, self.themes[name])
            
        return theme
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> None:
        """Deep merge overlay dict into base dict."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def apply_theme(self, name: str) -> None:
        """Apply a theme by name to the application."""
        theme = self.get_theme(name)
        self.current_theme = name
        
        # Apply theme colors to streamlit
        self._apply_theme_colors(theme)
        
        # Apply CSS variables for custom components
        self._inject_css_variables(theme)
    
    def _apply_theme_colors(self, theme: Dict[str, Any]) -> None:
        """Apply theme colors to Streamlit config."""
        colors = theme.get("colors", {})
        # This would integrate with Streamlit's theming if applicable
        # For now, store in session state for custom components
        st.session_state.theme_colors = colors
    
    def _inject_css_variables(self, theme: Dict[str, Any]) -> None:
        """Inject CSS variables based on theme into the page."""
        css = ":root {\n"
        
        # Add color variables
        for key, value in theme.get("colors", {}).items():
            css += f"  --color-{key}: {value};\n"
        
        # Add font variables
        for key, value in theme.get("fonts", {}).items():
            css += f"  --font-{key}: {value};\n"
            
        # Add setting variables
        for key, value in theme.get("settings", {}).items():
            css += f"  --{key}: {value};\n"
            
        css += "}\n"
        
        # Inject CSS into Streamlit
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    @staticmethod
    def toggle_theme():
        """Toggle between light and dark themes.
        
        This is a static convenience method that uses ThemeState to toggle themes.
        """
        logger.debug("Toggling theme via static method")
        
        # Toggle the theme mode through ThemeState
        new_mode = ThemeState.toggle_theme_mode()
        logger.debug(f"Theme toggled to: {new_mode}")
        
        # Get a ThemeManager instance to apply the theme
        theme_manager = ThemeManager()
        theme_manager.apply_theme(new_mode)
        
        return new_mode
    
    def get_available_themes(self) -> Dict[str, str]:
        """Get a dictionary of available themes with name and display name."""
        return {name: data.get("display_name", name.capitalize()) 
                for name, data in self.themes.items()}

def apply_custom_css():
    css = """
    <style>
    /* Your custom CSS here */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color-primary);
    }
    /* More styles... */
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)