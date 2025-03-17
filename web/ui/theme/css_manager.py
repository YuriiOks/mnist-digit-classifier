# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/theme/css_manager.py
# Description: Centralized CSS loading and management
# Created: 2025-03-17

import logging
import streamlit as st
from typing import List, Dict, Set, Optional

from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils


class CSSManager:
    """
    Centralized CSS loading and management.
    
    This class provides a way to load and inject CSS files
    in a consistent manner, with caching to prevent
    duplicate loading.
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CSSManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the CSS manager."""
        # Skip initialization if already done
        if getattr(self, '_initialized', False):
            return
            
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Track loaded CSS files to prevent duplicates
        self._loaded_files: Set[str] = set()
        
        # Cache CSS content by file path
        self._css_cache: Dict[str, str] = {}
        
        # Core CSS files that should be loaded for all views
        self._core_css_files = [
            # Global styles
            "global/variables.css",
            "global/animations.css",
            "global/base.css", 
            "global/reset.css",
            
            # Theme system
            "themes/theme-system.css",
            
            # Theme-aware component styles
            "components/theme_aware.css",
        ]
        
        # Common component CSS files
        self._common_component_css_files = [
            # Layout components
            "components/layout/header.css",
            "components/layout/footer.css",
            "components/layout/sidebar.css",
            
            # Common controls
            "components/controls/buttons.css",
            "components/controls/bb8-toggle.css",
            
            # Cards
            "components/cards/cards.css",
        ]
        
        # View-specific CSS files
        self._view_css_files = {
            "home": ["views/home.css"],
            "draw": ["views/draw.css"],
            "history": ["views/history.css"],
            "settings": ["views/settings.css"],
        }
        
        # Always load view_styles.css for all views
        self._common_view_css = ["views/view_styles.css"]
        
        self._initialized = True
        self._logger.debug("CSS Manager initialized")
    
    @AspectUtils.catch_errors
    def load_core_css(self) -> None:
        """Load core CSS files required for the application."""
        self._logger.debug("Loading core CSS files")
        self._load_css_files(self._core_css_files)
    
    @AspectUtils.catch_errors
    def load_common_component_css(self) -> None:
        """Load common component CSS files."""
        self._logger.debug("Loading common component CSS files")
        self._load_css_files(self._common_component_css_files)
    
    @AspectUtils.catch_errors
    def load_view_css(self, view_id: str) -> None:
        """
        Load CSS files for a specific view.
        
        Args:
            view_id: ID of the view to load CSS for
        """
        self._logger.debug(f"Loading CSS for view: {view_id}")
        
        # Always load the common view CSS
        self._load_css_files(self._common_view_css)
        
        # Load view-specific CSS if available
        if view_id in self._view_css_files:
            self._load_css_files(self._view_css_files[view_id])
        else:
            self._logger.warning(f"No specific CSS defined for view: {view_id}")
    
    @AspectUtils.catch_errors
    def _load_css_files(self, css_files: List[str]) -> None:
        """
        Load and inject multiple CSS files.
        
        Args:
            css_files: List of CSS file paths to load
        """
        combined_css = ""
        loaded_count = 0
        
        for css_file in css_files:
            # Skip already loaded files
            if css_file in self._loaded_files:
                self._logger.debug(f"CSS already loaded, skipping: {css_file}")
                continue
                
            # Try to get CSS from cache
            if css_file in self._css_cache:
                css_content = self._css_cache[css_file]
                self._logger.debug(f"Got CSS from cache: {css_file}")
            else:
                # Load CSS content
                css_content = resource_manager.load_css(css_file)
                
                # Cache CSS content if loaded successfully
                if css_content:
                    self._css_cache[css_file] = css_content
                    self._logger.debug(f"Loaded and cached CSS: {css_file}")
            
            # Add to combined CSS if loaded
            if css_content:
                combined_css += f"\n/* {css_file} */\n{css_content}"
                self._loaded_files.add(css_file)
                loaded_count += 1
            else:
                self._logger.warning(f"Failed to load CSS: {css_file}")
        
        # Inject combined CSS
        if combined_css:
            resource_manager.inject_css(combined_css)
            self._logger.debug(f"Injected {loaded_count} CSS files")
    
    @AspectUtils.catch_errors
    def load_single_css(self, css_file: str) -> None:
        """
        Load and inject a single CSS file.
        
        Args:
            css_file: CSS file path to load
        """
        self._load_css_files([css_file])
    
    @AspectUtils.catch_errors
    def clear_loaded_cache(self) -> None:
        """Clear the loaded files cache but keep content cache."""
        self._loaded_files = set()
        self._logger.debug("Cleared loaded files cache")
    
    @AspectUtils.catch_errors
    def clear_all_cache(self) -> None:
        """Clear all caches (loaded files and content)."""
        self._loaded_files = set()
        self._css_cache = {}
        self._logger.debug("Cleared all CSS caches")
    
    @property
    def loaded_files(self) -> List[str]:
        """Get list of loaded CSS files."""
        return list(self._loaded_files)


# Create a singleton instance
css_manager = CSSManager()