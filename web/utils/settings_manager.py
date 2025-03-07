import json
import os
import streamlit as st
from utils.resource_loader import ResourceLoader
from utils.theme_manager import ThemeManager

class SettingsManager:
    """Manager for application settings."""
    
    @staticmethod
    def load_config(config_name):
        """Load a configuration file from the settings config directory.
        
        Args:
            config_name: Name of the config file (without path or extension)
            
        Returns:
            Dict containing the configuration
        """
        try:
            config_path = os.path.join(
                ResourceLoader.get_app_dir(),
                "static",
                "config",
                "settings",
                f"{config_name}.json"
            )
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading settings config {config_name}: {str(e)}")
            return {}
    
    @staticmethod
    def get_theme_settings():
        """Get theme settings based on current theme state."""
        config = SettingsManager.load_config("theme_settings")
        is_dark = st.session_state.dark_mode
        
        return {
            "title": config.get("title", "Theme Settings"),
            "icon": config.get("icon", "🎨"),
            "current_theme": config.get("options", {}).get(
                "dark" if is_dark else "light", {}
            ).get("name", "Dark Mode" if is_dark else "Light Mode"),
            "checkbox_state": "checked" if is_dark else "",
            "toggle_label": config.get("options", {}).get(
                "dark" if is_dark else "light", {}
            ).get("label", "Switch to Light Mode" if is_dark else "Switch to Dark Mode")
        }
    
    @staticmethod
    def get_canvas_settings():
        """Get canvas settings configuration."""
        return SettingsManager.load_config("canvas_settings")
    
    @staticmethod
    def get_app_info():
        """Get application info configuration."""
        return SettingsManager.load_config("app_info")
    
    @staticmethod
    def reset_canvas_to_defaults():
        """Reset canvas settings to default values."""
        config = SettingsManager.get_canvas_settings()
        defaults = config.get("defaults", {})
        
        st.session_state.brush_size = defaults.get("brushSize", 20)
        st.session_state.brush_color = defaults.get("brushColor", "#000000") 