# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/navigation_state.py
# Description: Manages navigation-related application state
# Created: 2024-05-01

from typing import Dict, Any, List, Optional
import logging
import streamlit as st

from core.app_state.session_state import SessionState

# Create module-level logger
logger = logging.getLogger(__name__)

class NavigationState:
    """Manages navigation-related application state."""
    
    # State keys - use underscore prefix to avoid widget conflicts
    ACTIVE_VIEW_KEY = "_nav_active_view"
    NAV_CONFIG_KEY = "_nav_config"
    NAV_HISTORY_KEY = "_nav_history"
    
    # Default navigation config
    DEFAULT_NAV_CONFIG = {
        "items": [
            {
                "id": "home",
                "label": "Home",
                "icon": "🏠",
                "order": 1
            },
            {
                "id": "draw",
                "label": "Draw",
                "icon": "✏️",
                "order": 2
            },
            {
                "id": "history",
                "label": "History",
                "icon": "📊",
                "order": 3
            },
            {
                "id": "settings",
                "label": "Settings",
                "icon": "⚙️",
                "order": 4
            }
        ]
    }
    
    # Default active view
    DEFAULT_ACTIVE_VIEW = "home"
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize navigation state."""
        if "active_view" not in st.session_state:
            logger.debug("Initializing navigation state with default view")
            st.session_state["active_view"] = "home"
        
        # Set default nav config if not already set
        if not SessionState.has_key(cls.NAV_CONFIG_KEY):
            logger.debug(f"Setting default navigation config: {len(cls.DEFAULT_NAV_CONFIG['items'])} items")
            SessionState.set(cls.NAV_CONFIG_KEY, cls.DEFAULT_NAV_CONFIG)
        
        # Initialize navigation history if not already initialized
        if not SessionState.has_key(cls.NAV_HISTORY_KEY):
            logger.debug(f"Initializing navigation history with: {cls.DEFAULT_ACTIVE_VIEW}")
            SessionState.set(cls.NAV_HISTORY_KEY, [cls.DEFAULT_ACTIVE_VIEW])
            
        logger.debug("Navigation state initialization complete")
    
    @classmethod
    def get_active_view(cls) -> str:
        """Get the active view ID.
        
        Returns:
            str: Active view ID.
        """
        cls.initialize()
        return st.session_state["active_view"]
    
    @classmethod
    def set_active_view(cls, view_id: str) -> None:
        """Set the active view.
        
        Args:
            view_id: ID of the view to activate.
        """
        logger.debug(f"Setting active view: {view_id}")
        st.session_state["active_view"] = view_id
    
    @classmethod
    def get_routes(cls) -> List[Dict[str, Any]]:
        """Get the list of navigation routes."""
        logger.debug("Getting navigation routes")
        try:
            cls.initialize()
            nav_config = SessionState.get(cls.NAV_CONFIG_KEY, cls.DEFAULT_NAV_CONFIG)
            
            # Get routes from config and sort by order
            routes = nav_config.get("items", [])
            sorted_routes = sorted(routes, key=lambda x: x.get("order", 99))
            
            logger.debug(f"Retrieved {len(sorted_routes)} navigation routes")
            return sorted_routes
        except Exception as e:
            logger.error(f"Error getting navigation routes: {str(e)}", exc_info=True)
            return cls.DEFAULT_NAV_CONFIG.get("items", [])
    
    @classmethod
    def add_to_history(cls, view_id: str) -> None:
        """Add a view to navigation history."""
        logger.debug(f"Adding view {view_id} to navigation history")
        try:
            history = SessionState.get(cls.NAV_HISTORY_KEY, [])
            
            # Add view to history if it's not the same as the last entry
            if not history or history[-1] != view_id:
                history.append(view_id)
                # Limit history length to prevent unbounded growth
                if len(history) > 10:
                    history = history[-10:]
                SessionState.set(cls.NAV_HISTORY_KEY, history)
                logger.debug(f"Updated navigation history: {history}")
        except Exception as e:
            logger.error(f"Error adding view {view_id} to history: {str(e)}", exc_info=True)
            # Don't raise the exception since this is not critical functionality
    
    @classmethod
    def get_nav_history(cls) -> List[str]:
        """Get the navigation history.
        
        Returns:
            List[str]: Navigation history
        """
        logger.debug("Getting navigation history")
        try:
            cls.initialize()
            history = SessionState.get(cls.NAV_HISTORY_KEY, [])
            logger.debug(f"Retrieved navigation history with {len(history)} entries")
            return history
        except Exception as e:
            logger.error(f"Error retrieving navigation history: {str(e)}", exc_info=True)
            return []
    
    @classmethod
    def get_previous_view(cls) -> Optional[str]:
        """Get the previous view from navigation history.
        
        Returns:
            Optional[str]: ID of the previous view, or None if no previous view
        """
        logger.debug("Getting previous view from history")
        try:
            history = cls.get_nav_history()
            
            if len(history) >= 2:
                previous_view = history[-2]
                logger.debug(f"Previous view identified: {previous_view}")
                return previous_view
            
            logger.debug("No previous view found in navigation history")
            return None
        except Exception as e:
            logger.error(f"Error retrieving previous view: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def navigate_back(cls) -> bool:
        """Navigate to the previous view.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        logger.info("Attempting to navigate back to previous view")
        try:
            previous_view = cls.get_previous_view()
            
            if previous_view:
                # Remove current view from history
                history = cls.get_nav_history()
                history.pop()
                SessionState.set(cls.NAV_HISTORY_KEY, history)
                logger.debug(f"Removed current view from history")
                
                # Set active view to previous view
                SessionState.set(cls.ACTIVE_VIEW_KEY, previous_view)
                logger.info(f"Successfully navigated back to {previous_view}")
                return True
            
            logger.debug("Cannot navigate back: no previous view available")
            return False
        except Exception as e:
            logger.error(f"Error navigating back: {str(e)}", exc_info=True)
            return False