# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/css/view_css_loader.py
# Description: Utility for loading view-specific CSS
# Created: 2024-05-01

import streamlit as st
from typing import List, Optional
import logging

# Import the necessary functions from css_loader
from utils.css.css_loader import load_css_file, CSSLoadError, inject_css

logger = logging.getLogger(__name__)

def load_view_css(view_id: Optional[str] = None) -> None:
    """Load all CSS files for views and inject them.
    
    This function loads common view CSS and specific CSS for a view if provided.
    
    Args:
        view_id: Optional ID of the specific view to load CSS for.
               If not provided, only common view CSS is loaded.
    """
    try:
        logger.debug(f"Loading view CSS for: {view_id or 'common'}")
        
        # Load common view styles
        try:
            common_css = load_css_file("components/views/views.css")
            inject_css(common_css)
            logger.debug("Loaded common view CSS")
        except CSSLoadError as e:
            logger.warning(f"Could not load common view CSS: {str(e)}")
        
        # Load specific view styles if a view_id is provided
        if view_id:
            try:
                view_css = load_css_file(f"views/{view_id}.css")
                inject_css(view_css)
                logger.debug(f"Loaded specific view CSS for: {view_id}")
            except CSSLoadError as e:
                # Specific view CSS is optional
                logger.warning(f"Could not load specific view CSS for {view_id}: {str(e)}")
        
        logger.debug("View CSS loading complete")
    except Exception as e:
        logger.error(f"Error loading view CSS: {str(e)}", exc_info=True)
        raise CSSLoadError(f"Error loading view CSS: {str(e)}") 

def has_view_css(view_id: str) -> bool:
    """Check if a view has a specific CSS file.
    
    Args:
        view_id: ID of the view (e.g., "home", "settings").
        
    Returns:
        bool: True if the view has a CSS file, False otherwise.
    """
    logger.debug(f"Checking if view has CSS: {view_id}")
    try:
        css_path = f"assets/css/views/{view_id}.css"
        path = resolve_path(css_path)
        exists = path.exists()
        logger.debug(f"View CSS check for {view_id}: {'Found' if exists else 'Not found'}")
        return exists
    except Exception as e:
        logger.error(f"Error checking for view CSS {view_id}: {str(e)}")
        return False 