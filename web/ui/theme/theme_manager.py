# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/theme/theme_manager.py
# Description: Theme management for the application
# Created: 2024-05-01

import streamlit as st
import json
from typing import Dict, Any, Literal, Optional
import logging

from core.app_state.theme_state import ThemeState
from utils.file.path_utils import get_project_root
from utils.css.css_loader import load_css, load_theme_css, inject_css, load_css_file
from utils.js.js_loader import load_js

logger = logging.getLogger(__name__)

ThemeType = Literal["light", "dark"]

class ThemeManager:
    """Manager for application theming.
    
    Handles theme switching, theme-specific styling, and theme utilities.
    """
    
    # Theme options
    LIGHT_THEME = "light"
    DARK_THEME = "dark"
    
    # Theme configuration files
    _config_path = "assets/config/themes"
    _default_config = "default.json"
    _light_config = "light.json"
    _dark_config = "dark.json"
    
    # Cache for theme configurations
    _theme_cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the theme manager."""
        logger.info("Initializing ThemeManager")
        # Set default theme if not already set
        if "theme" not in st.session_state:
            st.session_state["theme"] = "light"
        
        # Apply theme on initialization
        cls.apply_theme()
        logger.info(f"Theme initialized: {cls.get_theme()}")
    
    def __init__(self):
        """Initialize theme manager."""
        logger.debug("Initializing ThemeManager")
        try:
            # Ensure theme state is initialized
            if not ThemeState.is_initialized():
                logger.debug("Theme state not initialized, initializing now")
                ThemeState.initialize()
                
            # Load theme configurations if not already loaded
            if not self._theme_cache:
                logger.debug("Theme cache empty, loading theme configurations")
                self._load_theme_configs()
                
            logger.debug("ThemeManager initialization complete")
        except Exception as e:
            logger.error(
                f"Error initializing ThemeManager: {str(e)}", 
                exc_info=True
            )
            raise
    
    @classmethod
    def _load_theme_configs(cls) -> None:
        """Load theme configurations from JSON files."""
        logger.debug("Loading theme configurations")
        try:
            root_path = get_project_root()
            logger.debug(f"Project root path: {root_path}")
            
            # Load default config
            default_path = f"{root_path}/{cls._config_path}/{cls._default_config}"
            logger.debug(f"Loading default theme config from: {default_path}")
            with open(default_path, 'r') as f:
                cls._theme_cache["default"] = json.load(f)
            
            # Load light theme config
            light_path = f"{root_path}/{cls._config_path}/{cls._light_config}"
            logger.debug(f"Loading light theme config from: {light_path}")
            with open(light_path, 'r') as f:
                cls._theme_cache[cls.LIGHT_THEME] = json.load(f)
            
            # Load dark theme config
            dark_path = f"{root_path}/{cls._config_path}/{cls._dark_config}"
            logger.debug(f"Loading dark theme config from: {dark_path}")
            with open(dark_path, 'r') as f:
                cls._theme_cache[cls.DARK_THEME] = json.load(f)
                
            logger.info(f"Successfully loaded theme configurations: {list(cls._theme_cache.keys())}")
        except Exception as e:
            logger.error(f"Error loading theme configurations: {str(e)}", exc_info=True)
            # Initialize with empty configs as fallback
            logger.warning("Initializing with empty theme configurations as fallback")
            if "default" not in cls._theme_cache:
                cls._theme_cache["default"] = {}
            if cls.LIGHT_THEME not in cls._theme_cache:
                cls._theme_cache[cls.LIGHT_THEME] = {}
            if cls.DARK_THEME not in cls._theme_cache:
                cls._theme_cache[cls.DARK_THEME] = {}
    
    @classmethod
    def get_theme(cls) -> ThemeType:
        """Get the current theme.
        
        Returns:
            ThemeType: Current theme ("light" or "dark").
        """
        return st.session_state.get("theme", "light")
    
    @classmethod
    def set_theme(cls, theme: ThemeType) -> None:
        """Set the theme.
        
        Args:
            theme: Theme to set ("light" or "dark").
        """
        logger.debug(f"Setting theme to: {theme}")
        st.session_state["theme"] = theme
        cls.apply_theme()
    
    @classmethod
    def toggle_theme(cls) -> None:
        """Toggle between light and dark themes."""
        current_theme = cls.get_theme()
        new_theme = "dark" if current_theme == "light" else "light"
        logger.debug(f"Toggling theme from {current_theme} to {new_theme}")
        cls.set_theme(new_theme)
    
    @classmethod
    def is_dark_mode(cls) -> bool:
        """Check if dark mode is active.
        
        Returns:
            bool: True if dark mode is active, False otherwise
        """
        logger.debug("Checking if dark mode is active")
        try:
            is_dark = cls.get_theme() == cls.DARK_THEME
            logger.debug(f"Dark mode active: {is_dark}")
            return is_dark
        except Exception as e:
            logger.error(f"Error checking dark mode: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def get_theme_color(cls, color_name: str, for_inline: bool = False) -> str:
        """Get color value from current theme.
        
        Args:
            color_name: Color name from theme configuration
            for_inline: Whether to return CSS variable (False) or actual value (True)
            
        Returns:
            str: Color value or CSS variable
        """
        logger.debug(f"Getting theme color: {color_name} (for_inline={for_inline})")
        try:
            if for_inline:
                theme = cls.get_theme()
                if theme not in cls._theme_cache:
                    logger.debug(f"Theme {theme} not in cache, loading theme configs")
                    cls._load_theme_configs()
                
                if (
                    theme in cls._theme_cache and 
                    "colors" in cls._theme_cache[theme] and 
                    color_name in cls._theme_cache[theme]["colors"]
                ):
                    color = cls._theme_cache[theme]["colors"][color_name]
                    logger.debug(f"Found color value: {color}")
                    return color
                logger.warning(f"Color {color_name} not found in theme {theme}, using fallback")
                return "#000000"  # Fallback color
            else:
                css_var = f"var(--{color_name}-color)"
                logger.debug(f"Returning CSS variable: {css_var}")
                return css_var
        except Exception as e:
            logger.error(f"Error getting theme color {color_name}: {str(e)}", exc_info=True)
            return "#000000" if for_inline else f"var(--{color_name}-color)"
    
    @classmethod
    def get_theme_config(cls, key: str, default: Any = None) -> Any:
        """Get configuration value from current theme.
        
        Args:
            key: Configuration key path (dot notation)
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        theme = cls.get_theme()
        if theme not in cls._theme_cache:
            cls._load_theme_configs()
        
        # Handle dot notation for nested configs
        parts = key.split('.')
        config = cls._theme_cache.get(theme, {})
        
        for part in parts:
            if part in config:
                config = config[part]
            else:
                return default
        
        return config
    
    @classmethod
    def get_font(cls, font_type: str = "body") -> str:
        """Get font family based on type.
        
        Args:
            font_type: Font type (body, heading, code)
            
        Returns:
            str: Font family value
        """
        if "default" not in cls._theme_cache:
            cls._load_theme_configs()
            
        if (
            "default" in cls._theme_cache and 
            "fonts" in cls._theme_cache["default"] and 
            font_type in cls._theme_cache["default"]["fonts"]
        ):
            return cls._theme_cache["default"]["fonts"][font_type]
        
        # Fallback fonts
        if font_type == "heading":
            return "system-ui, sans-serif"
        elif font_type == "code":
            return "monospace"
        else:
            return "system-ui, sans-serif"
    
    @classmethod
    def get_theme_icon(cls, icon_name: str) -> str:
        """Get theme-specific icon path.
        
        Args:
            icon_name: Icon name without extension
            
        Returns:
            str: Path to icon for current theme
        """
        theme = cls.get_theme()
        return f"assets/images/icons/{theme}/{icon_name}.svg"
    
    @classmethod
    def inject_theme_css(cls) -> None:
        """Inject theme-specific CSS."""
        # Load global CSS first
        load_css("assets/css/global/variables.css")
        load_css("assets/css/global/reset.css")
        load_css("assets/css/global/typography.css")
        
        # Load theme-specific variables
        theme = cls.get_theme()
        load_css(f"assets/css/themes/{theme}/variables.css")
        
        # Add transition CSS for smooth theme switching
        transition_css = """
        <style>
        body, * {
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease,
                        box-shadow 0.3s ease, fill 0.3s ease, stroke 0.3s ease;
        }
        </style>
        """
        st.markdown(transition_css, unsafe_allow_html=True)
    
    @classmethod
    def inject_theme_scripts(cls) -> None:
        """Inject theme-related JavaScript."""
        # Load theme detector script to detect system preferences
        load_js("assets/js/theme/theme_detector.js")
        
        # Load theme toggle script
        load_js("assets/js/theme/theme_toggle.js")
    
    @classmethod
    def apply_theme_js(cls) -> None:
        """Apply theme changes using JavaScript for a more responsive experience."""
        logger.debug("Applying theme via JavaScript")
        try:
            current_theme = cls.get_theme()
            logger.debug(f"Applying theme via JavaScript: {current_theme}")
            
            # JavaScript to update theme without page refresh
            js_code = f"""
            <script>
                (function() {{
                    // Apply theme immediately
                    document.documentElement.setAttribute('data-theme', '{current_theme}');
                    
                    // Handle body class for dark theme
                    if ('{current_theme}' === 'dark') {{
                        document.body.classList.add('dark');
                    }} else {{
                        document.body.classList.remove('dark');
                    }}
                    
                    console.log('Theme updated via JS: {current_theme}');
                }})();
            </script>
            """
            st.markdown(js_code, unsafe_allow_html=True)
            logger.debug("Theme JS applied successfully")
        except Exception as e:
            logger.error(f"Error applying theme JS: {str(e)}", exc_info=True)
            # Don't raise, allow application to continue
    
    @classmethod
    def initialize_theme_system(cls) -> None:
        """Initialize the theme system.
        
        This method should be called at application startup to ensure
        theme state is properly initialized.
        """
        logger.debug("Initializing theme system")
        try:
            # Initialize theme state
            if not ThemeState.is_initialized():
                logger.debug("Theme state not initialized, initializing now")
                ThemeState.initialize()
            
            # Load theme configurations
            if not cls._theme_cache:
                logger.debug("Theme cache empty, loading theme configurations")
                cls._load_theme_configs()
            
            # Load theme CSS
            logger.debug("Loading theme CSS")
            theme = cls.get_theme()
            load_theme_css(theme)
            
            # Load theme JavaScript
            logger.debug("Loading theme JavaScript")
            theme_toggle_js = "assets/js/theme/theme_toggle.js"
            theme_detector_js = "assets/js/theme/theme_detector.js"
            load_js(theme_toggle_js)
            load_js(theme_detector_js)
            
            logger.info("Theme system initialization complete")
        except Exception as e:
            logger.error(f"Error initializing theme system: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def detect_system_preference(cls) -> None:
        """Detect and apply system color scheme preference."""
        js_code = """
        <script>
            (function() {
                if (window.detectSystemThemePreference) {
                    window.detectSystemThemePreference();
                }
            })();
        </script>
        """
        st.markdown(js_code, unsafe_allow_html=True)
    
    @classmethod
    def get_contrast_color(cls, background_color: str) -> str:
        """Get contrasting color (light or dark) for given background.
        
        Args:
            background_color: Background color to contrast against
            
        Returns:
            str: Contrasting color (white or black)
        """
        # Simple algorithm to determine if text should be light or dark
        # This is a simplified version; more sophisticated algorithms exist
        if background_color.startswith('#'):
            # Convert hex to RGB
            r = int(background_color[1:3], 16)
            g = int(background_color[3:5], 16)
            b = int(background_color[5:7], 16)
            
            # Calculate luminance using the formula (0.299*R + 0.587*G + 0.114*B)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Return white for dark backgrounds, black for light backgrounds
            return "#ffffff" if luminance < 128 else "#000000"
        
        # Fallback for non-hex colors
        return "#ffffff" if cls.is_dark_mode() else "#000000" 
    
    @classmethod
    def apply_theme(cls) -> None:
        """Apply the current theme by injecting theme CSS and setting attributes."""
        theme_name = cls.get_theme()
        logger.debug(f"Applying theme: {theme_name}")
        
        try:
            # Load theme CSS
            theme_css = load_theme_css(theme_name)
            
            # Apply theme attribute to body to enable CSS selectors
            theme_attribute = f"""
            <script>
                (function() {{
                    document.body.setAttribute('data-theme', '{theme_name}');
                    console.log('Theme applied:', '{theme_name}');
                }})();
            </script>
            """
            
            # Inject theme CSS and attribute
            inject_css(theme_css)
            st.markdown(theme_attribute, unsafe_allow_html=True)
            
            # Add additional optimization for Streamlit elements
            streamlit_theme_fixes = f"""
            <style>
                /* Apply theme to Streamlit components */
                .stApp {{
                    background-color: var(--color-background);
                    color: var(--color-text);
                }}
                
                /* Style for buttons to match our theme */
                .stButton button {{
                    background-color: var(--color-primary);
                    color: white;
                    border-radius: var(--border-radius-md);
                    transition: all var(--transition-speed-fast) var(--transition-timing);
                }}
                
                .stButton button:hover {{
                    background-color: var(--color-primary-dark);
                    box-shadow: var(--shadow-md);
                }}
                
                /* Make sure content area has the right background */
                .main-content {{
                    background-color: var(--color-background);
                    padding: var(--spacing-md);
                    border-radius: var(--border-radius-md);
                }}
            </style>
            """
            st.markdown(streamlit_theme_fixes, unsafe_allow_html=True)
            
            logger.debug(f"Theme {theme_name} applied successfully")
        except Exception as e:
            logger.error(f"Error applying theme {theme_name}: {str(e)}", exc_info=True)
    
    @classmethod
    def inject_card_styles(cls) -> None:
        """Inject CSS styles for cards."""
        logger.debug("Injecting card styles")
        try:
            # Load global card styles
            try:
                card_css = load_css_file("assets/css/global/cards.css")
                inject_css(card_css)
                logger.debug("Injected global card styles")
            except Exception as e:
                logger.warning(f"Failed to load global card styles: {str(e)}")
                
                # Fallback to inline styles if file loading fails
                fallback_css = """
                .card, .content-card {
                  background-color: white;
                  border: 1px solid #ddd;
                  border-radius: 0.5rem;
                  padding: 1.5rem;
                  margin-bottom: 1.5rem;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .card-title, .content-card .card-title {
                  font-size: 1.25rem;
                  font-weight: 600;
                  margin-bottom: 1rem;
                  color: #333;
                }
                
                .card-content {
                  color: #555;
                  line-height: 1.5;
                }
                
                .card-icon {
                  margin-right: 0.5rem;
                }
                
                [data-theme="dark"] .card, [data-theme="dark"] .content-card {
                  background-color: #2a2a2a;
                  border-color: #444;
                  color: #eee;
                }
                
                [data-theme="dark"] .card-title, [data-theme="dark"] .content-card .card-title {
                  color: #eee;
                }
                
                [data-theme="dark"] .card-content {
                  color: #ccc;
                }
                """
                inject_css(fallback_css)
                logger.debug("Injected fallback card styles")
            
            # Also load component-specific styles for Cards
            try:
                component_card_css = load_css_file("assets/css/components/cards/card.css")
                content_card_css = load_css_file("assets/css/components/cards/content_card.css")
                inject_css(component_card_css)
                inject_css(content_card_css)
                logger.debug("Injected component-specific card styles")
            except Exception as e:
                logger.warning(f"Failed to load component card styles: {str(e)}")
                
            logger.debug("Card styles injected successfully")
        except Exception as e:
            logger.error(f"Error injecting card styles: {str(e)}", exc_info=True)

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