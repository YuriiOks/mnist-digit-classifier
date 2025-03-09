# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/__init__.py
# Description: App state management package
# Created: 2024-05-01

"""State management for the MNIST Digit Classifier application."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import for easier access
from core.app_state.session_state import SessionState

# Check if these modules exist before importing them
try:
    from core.app_state.canvas_state import CanvasState
except ImportError:
    # Define a placeholder to avoid errors
    class CanvasState:
        """Placeholder for CanvasState"""
        @classmethod
        def initialize(cls): pass

try:
    from core.app_state.history_state import HistoryState
except ImportError:
    # Define a placeholder to avoid errors
    class HistoryState:
        """Placeholder for HistoryState"""
        @classmethod
        def initialize(cls): pass

try:
    from core.app_state.settings_state import SettingsState
except ImportError:
    # Define a placeholder to avoid errors
    class SettingsState:
        """Placeholder for SettingsState"""
        @classmethod
        def initialize(cls): pass
        
        @classmethod
        def get_setting(cls, category, key, default=None):
            return default

try:
    from core.app_state.theme_state import ThemeState
except ImportError:
    # Define a placeholder to avoid errors
    class ThemeState:
        """Placeholder for ThemeState"""
        @classmethod
        def initialize(cls): pass

try:
    from core.app_state.navigation_state import NavigationState
except ImportError:
    # Define a placeholder to avoid errors
    class NavigationState:
        """Placeholder for NavigationState"""
        @classmethod
        def initialize(cls): pass

__all__ = [
    'SessionState',
    'CanvasState',
    'HistoryState',
    'SettingsState',
    'ThemeState',
    'NavigationState',
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
