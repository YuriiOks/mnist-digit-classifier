# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/__init__.py
# Description: App state package initialization
# Created: 2024-05-05

"""Application state management for the MNIST Digit Classifier."""

from typing import Dict, Any
from core.app_state.session_state import SessionState
from core.app_state.theme_state import ThemeState
from core.app_state.navigation_state import NavigationState
from core.app_state.canvas_state import CanvasState
from core.app_state.history_state import HistoryState
from core.app_state.settings_state import SettingsState
# from core.app_state.home_state import HomeState
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "SessionState",
    "ThemeState",
    "NavigationState",
    "CanvasState",
    "HistoryState",
    "SettingsState",
    # "HomeState"
]

def initialize_app_state():
    """Initialize the application state."""
    logger.info("Initializing app state")
    
    # Initialize all state components
    SessionState.initialize()
    
    try:
        CanvasState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize CanvasState: {e}")
    
    try:
        HistoryState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize HistoryState: {e}")
    
    try:
        SettingsState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize SettingsState: {e}")
    
    try:
        ThemeState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize ThemeState: {e}")
    
    try:
        NavigationState.initialize()
    except Exception as e:
        logger.warning(f"Failed to initialize NavigationState: {e}")

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
        },
        "settings": {
            "theme": SettingsState.get_setting("theme", "mode"),
            "canvas": SettingsState.get_setting("canvas", "size"),
            "prediction": SettingsState.get_setting("prediction", "auto_predict")
        }
    } 
