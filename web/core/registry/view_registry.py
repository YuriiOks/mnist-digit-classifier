# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/registry/view_registry.py
# Description: Registry for application views
# Created: 2024-05-01

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ViewRegistry:
    """Registry for application views.
    
    This class keeps track of all views in the application and allows
    retrieving them by ID.
    """
    
    # Class variable to store registered views
    _views = {}
    
    @classmethod
    def register_view(cls, view):
        """Register a view with the registry.
        
        Args:
            view: View instance to register.
        """
        if not hasattr(view, 'view_id'):
            logger.error(f"Cannot register view {view}: Missing view_id attribute")
            return
            
        view_id = view.view_id
        logger.debug(f"Registering view: {view_id}")
        cls._views[view_id] = view
        
    @classmethod
    def get_view(cls, view_id: str):
        """Get a view by its ID.
        
        Args:
            view_id: ID of the view to retrieve.
            
        Returns:
            View instance if found, None otherwise.
        """
        if view_id in cls._views:
            return cls._views[view_id]
        logger.warning(f"View not found: {view_id}")
        return None
        
    @classmethod
    def get_all_views(cls) -> List[Any]:
        """Get all registered views.
        
        Returns:
            List of all registered view instances.
        """
        return list(cls._views.values())
        
    @classmethod
    def get_view_ids(cls) -> List[str]:
        """Get IDs of all registered views.
        
        Returns:
            List of view IDs
        """
        return list(cls._views.keys()) 