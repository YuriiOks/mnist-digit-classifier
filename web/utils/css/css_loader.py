# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/css_loader.py
# Description: CSS loading utilities
# Created: 2024-05-01

import os
import logging
import streamlit as st

from utils.file.path_utils import get_project_root

logger = logging.getLogger(__name__)

class CSSLoadError(Exception):
    """Exception raised for errors in CSS loading operations."""
    def __init__(self, message, css_file=None):
        self.message = message
        self.css_file = css_file
        super().__init__(self.message)

def load_css_file(css_path: str) -> str:
    """Load CSS from a file.
    
    Args:
        css_path: Path to the CSS file, relative to project root.
        
    Returns:
        str: CSS content.
        
    Raises:
        CSSLoadError: If the CSS file doesn't exist or can't be loaded.
    """
    logger.debug(f"Loading CSS from {css_path}")
    full_path = os.path.join(get_project_root(), css_path)
    
    try:
        with open(full_path, "r") as f:
            content = f.read()
        logger.debug(f"CSS loaded successfully: {len(content)} bytes")
        return content
    except FileNotFoundError:
        logger.error(f"CSS file not found: {full_path}")
        raise CSSLoadError(f"CSS file not found: {css_path}", css_file=css_path)
    except Exception as e:
        logger.error(f"Error loading CSS: {str(e)}")
        raise CSSLoadError(f"Error loading CSS from {css_path}: {str(e)}", css_file=css_path) from e

def load_component_css(component_type: str, component_name: str) -> str:
    """Load CSS for a component.
    
    Args:
        component_type: Type of component (e.g., "cards", "controls").
        component_name: Name of the component (e.g., "card", "button").
        
    Returns:
        str: CSS content.
        
    Raises:
        FileNotFoundError: If the CSS file doesn't exist.
    """
    css_path = f"assets/css/components/{component_type}/{component_name}.css"
    return load_css_file(css_path)

def inject_css(css_content: str) -> None:
    """Inject CSS into the Streamlit app.
    
    Args:
        css_content: CSS content to inject.
    """
    # Use the streamlit-specific way to inject CSS for best compatibility
    st.markdown(f"""
    <style>
    {css_content}
    </style>
    """, unsafe_allow_html=True)

def load_and_inject_css(css_path: str) -> None:
    """Load CSS from a file and inject it into the Streamlit app.
    
    Args:
        css_path: Path to the CSS file, relative to project root.
    """
    logger.debug(f"Loading and injecting CSS from: {css_path}")
    try:
        css_content = load_css_file(css_path)
        inject_css(css_content)
        logger.debug(f"Successfully loaded and injected CSS from: {css_path}")
    except Exception as e:
        logger.error(f"Failed to load and inject CSS {css_path}: {str(e)}")

def load_theme_css(theme_name: str = "light") -> str:
    """Load theme-specific CSS variables.
    
    Args:
        theme_name: Name of the theme (e.g., "light", "dark").
    
    Returns:
        str: The contents of the theme CSS file.
    
    Raises:
        CSSLoadError: If the theme CSS file doesn't exist.
    """
    logger.debug(f"Loading theme CSS for: {theme_name}")
    try:
        # Fix the path to include assets/css/
        css_path = f"assets/css/themes/{theme_name}/variables.css"
        logger.debug(f"Theme CSS path: {css_path}")
        return load_css_file(css_path)
    except Exception as e:
        logger.error(f"Failed to load theme CSS for {theme_name}: {str(e)}")
        raise CSSLoadError(f"Failed to load theme CSS for {theme_name}: {str(e)}", 
                          css_file=f"assets/css/themes/{theme_name}/variables.css") from e

def load_css(css_path: str) -> None:
    """Load and inject a CSS file.
    
    This is a convenience function that combines load_css_file and inject_css.
    
    Args:
        css_path: Path to the CSS file, relative to project root.
    """
    logger.debug(f"Loading CSS: {css_path}")
    try:
        css_content = load_css_file(css_path)
        inject_css(css_content)
        logger.debug(f"Successfully loaded CSS: {css_path}")
    except Exception as e:
        logger.warning(f"Failed to load CSS {css_path}: {str(e)}")

def load_component_css(component_type: str, component_name: str) -> None:
    """Load CSS for a specific component.
    
    Args:
        component_type: Type of component (e.g., "button", "card").
        component_name: Name of the component (e.g., "primary", "info").
    """
    logger.debug(f"Loading component CSS for: {component_type}/{component_name}")
    try:
        component_type = "layout" if component_type == "navigation" else component_type
        css_path = f"assets/css/components/{component_type}/{component_name}.css"
        load_css(css_path)
        logger.debug(f"Successfully loaded component CSS for: {component_type}/{component_name}")
    except Exception as e:
        logger.warning(f"Failed to load component CSS for {component_type}/{component_name}: {str(e)}")

def ensure_css_loaded(theme_name: str = "light"):
    """Ensure all required CSS is loaded for the application."""
    logger.debug(f"Ensuring CSS is loaded for theme: {theme_name}")
    try:
        # 1. Reset CSS (normalize browser styles)
        load_css("assets/css/global/reset.css")
        
        # 2. Load fonts FIRST so they're available early
        load_css("assets/css/global/fonts.css")
        
        # 3. Global CSS variables and base styles
        load_css("assets/css/global/variables.css")
        load_css("assets/css/global/typography.css")
        load_css("assets/css/global/grid.css")
        load_css("assets/css/global/cards.css")
        load_css("assets/css/global/streamlit_overrides.css")
        
        # 4. Theme-specific CSS
        css_content = load_theme_css(theme_name)
        inject_css(css_content)
        
        # 5. Component CSS
        load_css("assets/css/components/layout/header.css")
        load_css("assets/css/components/layout/footer.css")
        load_css("assets/css/components/layout/sidebar.css")
        load_css("assets/css/components/cards/card.css")
        load_css("assets/css/components/cards/content_card.css")
        
        # 6. View-specific CSS
        load_css("assets/css/views/home.css")
        
        logger.info("All CSS loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CSS: {str(e)}", exc_info=True)
        # Add fallback CSS to ensure basic styling works
        inject_fallback_css()

def inject_fallback_css():
    """Inject minimal CSS to ensure basic functionality if main CSS fails."""
    basic_css = """
    body { font-family: system-ui, sans-serif; }
    .card, .content-card { 
        border: 1px solid #ddd; 
        padding: 1rem; 
        margin-bottom: 1rem; 
        border-radius: 0.5rem;
    }
    .card-title { font-weight: bold; margin-bottom: 0.5rem; }
    """
    st.markdown(f"<style>{basic_css}</style>", unsafe_allow_html=True)

def debug_css():
    """Debug CSS loading issues."""
    logger.debug("Entering CSS debug mode")
    st.title("CSS Debug View")
    
    # Try to directly inject some test CSS to verify styling works
    logger.debug("Injecting test CSS")
    st.markdown("""
    <style>
    .test-box {
        background-color: red;
        padding: 20px;
        color: white;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    
    <div class="test-box">
        If this box is red with white text, basic CSS injection works
    </div>
    """, unsafe_allow_html=True)
    
    # List all CSS files that should be loaded
    st.write("### CSS Files that should be loaded:")
    css_files = [
        "assets/css/global/reset.css",
        "assets/css/global/variables.css",
        "assets/css/themes/light/variables.css",

    ]
    
    logger.debug(f"Checking existence of {len(css_files)} CSS files")
    
    # Check if files exist
    for css_file in css_files:
        try:
            from utils.file.path_utils import resolve_path
            path = resolve_path(css_file)
            exists = path.exists()
            st.write(f"- {css_file}: {'✅ Exists' if exists else '❌ Missing'}")
            logger.debug(f"CSS file check: {css_file} - {'Exists' if exists else 'Missing'}")
        except Exception as e:
            st.write(f"- {css_file}: ❌ Error: {str(e)}")
            logger.error(f"Error checking CSS file {css_file}: {str(e)}")
    
    logger.debug("CSS debug view rendered")

# def load_all_css():
#     """Load all CSS files."""
#     ensure_css_loaded()
