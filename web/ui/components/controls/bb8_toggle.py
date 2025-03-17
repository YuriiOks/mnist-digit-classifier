# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/bb8_toggle.py
# Description: BB8-themed toggle component for theme switching
# Created: 2025-03-16

import streamlit as st
from st_click_detector import click_detector
import logging
from typing import Optional, Callable, Dict, Any

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager, theme_manager
from utils.resource_manager import resource_manager
from utils.aspects import AspectUtils
from typing import Optional, Callable, Dict, Any

# In ui/components/controls/bb8_toggle.py

class BB8Toggle(Component[Dict[str, Any]]):
    """BB8-themed toggle component for theme switching."""

    def __init__(
        self,
        theme_manager_instance: Optional[ThemeManager] = None,
        on_change: Optional[Callable[[str], None]] = None,
        *,
        key: str = "bb8_toggle",
        id: Optional[str] = None,
        classes: Optional[list] = None,
        attributes: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Initialize the BB8Toggle component."""
        super().__init__(
            component_type="controls",
            component_name="bb8_toggle",
            id=id,
            classes=classes or [],
            attributes=attributes or {},
            key=key,
            **kwargs
        )

        self.__theme_manager = theme_manager_instance or theme_manager
        self.__on_change = on_change
        self.__key = key
    
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def _load_bb8_toggle_css(self) -> None:
        """Load the BB8 toggle CSS."""
        # Same as your existing implementation
        # Try multiple potential paths for the CSS
        potential_paths = [
            "components/controls/bb8-toggle.css",
            "components/controls/bb8_toggle.css",
            "controls/bb8-toggle.css",
            "controls/bb8_toggle.css"
        ]
        
        css_loaded = False
        for css_path in potential_paths:
            css_content = resource_manager.load_css(css_path)
            if css_content:
                resource_manager.inject_css(css_content)
                self._logger.debug(f"Successfully loaded BB8 toggle CSS from {css_path}")
                css_loaded = True
                break
        
        if not css_loaded:
            # If no CSS was loaded, inject the minimum CSS needed
            self._logger.warning("Could not load BB8 toggle CSS from files, using inline minimum CSS")
            min_css = """
            .bb8-toggle {
                display: inline-block;
                cursor: pointer;
            }
            /* Minimal styles to make it work */
            """
            resource_manager.inject_css(min_css)

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> str:
        """Render the BB8 toggle component."""
        # Get current theme
        current_theme = self.__theme_manager.get_current_theme()
        is_dark = self.__theme_manager.is_dark_mode()
        
        # Simple inline styled version (we're not using the template anymore)
        return f"""
        <a href="#" id="bb8-toggle">
            <!-- BB8 body -->
            <div style="
                position: absolute;
                left: {105 if not is_dark else 'calc(100% - 75px)'}px;
                top: 15px;
                transition: 0.4s;
                z-index: 2;
            ">
                <!-- Head -->
                <div style="
                    width: 40px;
                    height: 25px;
                    background: white;
                    border-radius: 25px 25px 0 0;
                    margin-bottom: -3px;
                "></div>
                
                <!-- Body -->
                <div style="
                    width: 60px;
                    height: 60px;
                    background: white;
                    border-radius: 50%;
                "></div>
            </div>
            
            <!-- Ground -->
            <div style="
                position: absolute; 
                width: 100%; 
                height: 30%; 
                bottom: 0; 
                background: #b18d71; 
                z-index: 1;
            "></div>
        </a>
        """

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> Dict[str, Any]:
        """Display the BB8 toggle with click detection."""
        try:
            from st_click_detector import click_detector
        except ImportError:
            st.error("Please install st-click-detector: pip install streamlit-click-detector")
            return {"error": "st_click_detector not installed"}
        
        # Get theme state
        current_theme = self.__theme_manager.get_current_theme()
        is_dark = current_theme == "dark"
        
        # Standalone BB8 toggle with ALL styling inline
        bb8_html = f"""
        <div style="text-align:center; padding-top:20px; padding-bottom:20px;">
        <a href="#" id="bb8-toggle" style="display:inline-block; text-decoration:none;">
            <div style="
            width: 170px;
            height: 90px;
            background: linear-gradient(#2c4770, #070e2b 35%, #628cac 50% 70%, #a6c5d4);
            background-position-y: {0 if is_dark else '-90px'};
            border-radius: 99em;
            position: relative;
            overflow: hidden;
            transition: 0.4s;
            cursor: pointer;
            ">
            <!-- BB8 droid -->
            <div style="
                position: absolute;
                left: {85 if is_dark else 15}px;
                top: 15px;
                transition: 0.4s;
                display: flex;
                flex-direction: column;
                align-items: center;
                z-index: 2;
            ">
                <!-- Head -->
                <div style="
                width: 40px;
                height: 25px;
                background: white;
                border-radius: 25px 25px 0 0;
                margin-bottom: -3px;
                position: relative;
                "></div>
                
                <!-- Body -->
                <div style="
                width: 60px;
                height: 60px;
                background: white;
                border-radius: 50%;
                position: relative;
                ">
                <!-- Orange marking -->
                <div style="
                    position: absolute;
                    width: 20px;
                    height: 20px;
                    background: #de7d2f;
                    border-radius: 50%;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                "></div>
                </div>
            </div>
            
            <!-- Ground -->
            <div style="
                position: absolute;
                width: 100%;
                height: 30%;
                bottom: 0;
                background: #b18d71;
                z-index: 1;
            "></div>
            
            <!-- Shadow -->
            <div style="
                position: absolute;
                width: 60px;
                height: 10px;
                background: rgba(0,0,0,0.2);
                border-radius: 50%;
                bottom: 10px;
                left: {85 if is_dark else 15}px;
                transform: skew({70 if is_dark else -70}deg);
                transition: 0.4s;
                z-index: 1;
            "></div>
            </div>
        </a>
        </div>
        """
        
        # Use click detector
        clicked = click_detector(bb8_html, key=f"bb8_{current_theme}")
        
        # Handle click
        if clicked == "bb8-toggle":
            new_theme = "light" if is_dark else "dark"
            self.__theme_manager.apply_theme(new_theme)
            
            # Call on_change if provided
            if self.__on_change:
                self.__on_change(new_theme)
            
            st.rerun()
        
        return {
            "theme": current_theme,
            "is_dark": is_dark,
            "key": self.__key
        }