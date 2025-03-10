# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/resource_loader.py
# Description: Resource loader utility for loading application resources
# Created: 2024-05-01

import logging
import streamlit as st
from typing import List, Optional, Union

from utils.css.css_loader import load_css, load_theme_css, inject_css, load_css_file
from utils.html.html_loader import load_html_template

logger = logging.getLogger(__name__)

class ResourceLoader:
    """Resource loading utility for the application.
    
    This class provides static methods to load various resources,
    including CSS, HTML templates, and other assets.
    """
    
    @staticmethod
    def load_css(css_files: List[str]) -> None:
        """Load and inject multiple CSS files.
        
        Args:
            css_files: List of CSS file paths to load
        """
        logger.debug(f"Loading {len(css_files)} CSS files")
        
        try:
            for css_file in css_files:
                try:
                    css_content = load_css_file(css_file)
                    inject_css(css_content)
                    logger.debug(f"Loaded CSS file: {css_file}")
                except Exception as e:
                    logger.warning(f"Could not load CSS file {css_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading CSS files: {str(e)}")
            
    @staticmethod
    def load_theme_css(theme: str = "light") -> None:
        """Load theme-specific CSS.
        
        Args:
            theme: Theme name (light or dark)
        """
        try:
            load_theme_css(theme)
            logger.debug(f"Loaded {theme} theme CSS")
        except Exception as e:
            logger.error(f"Error loading theme CSS: {str(e)}")
            
    @staticmethod
    def load_html_template(template_path: str) -> str:
        """Load an HTML template file.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            str: The HTML template content
        """
        try:
            return load_html_template(template_path)
        except Exception as e:
            logger.error(f"Error loading HTML template {template_path}: {str(e)}")
            return f"<div class='error'>Error loading template: {str(e)}</div>"
