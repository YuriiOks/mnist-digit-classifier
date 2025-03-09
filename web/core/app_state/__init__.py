# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/__init__.py
# Description: App state management
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any

from core.app_state.session_state import SessionState
from core.app_state.theme_state import ThemeState
from core.app_state.navigation_state import NavigationState
from core.app_state.canvas_state import CanvasState
from core.app_state.history_state import HistoryState

logger = logging.getLogger(__name__)

def initialize_app_state():
    """Initialize the application state."""
    logger.info("Initializing app state")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['active_view'] = 'home'
        st.session_state['theme'] = 'light'
        st.session_state['history'] = []
        logger.info("Session state initialized")
    else:
        logger.debug("Session state already initialized")
    
    return st.session_state

def get_app_state() -> Dict[str, Any]:
    """Get a dictionary containing all application state.
    
    Returns:
        Dict[str, Any]: Dictionary containing all application state.
    """
    return {
        "theme": {
            "current": ThemeState.get_current_theme(),
            "is_dark": ThemeState.get_current_theme() == ThemeState.THEME_DARK
        },
        "navigation": {
            "active_view": NavigationState.get_active_view(),
            "history": NavigationState.get_nav_history(),
            "routes": NavigationState.get_routes()
        },
        "canvas": {
            "image": CanvasState.get_image(),
            "prediction": CanvasState.get_prediction()
        },
        "history": {
            "predictions": HistoryState.get_predictions(),
            "latest": HistoryState.get_latest_prediction()
        }
    } 