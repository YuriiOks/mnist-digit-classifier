# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/layout/sidebar.py
# Description: Application sidebar with navigation
# Created: 2024-05-01

import streamlit as st
import logging
from typing import List, Dict, Any, Optional

from core.app_state.navigation_state import NavigationState
from core.app_state.theme_state import ThemeState
from ui.components.navigation.option_menu import create_option_menu
from ui.theme.theme_manager import ThemeManager
from ui.components.controls.bb8_toggle import BB8Toggle

logger = logging.getLogger(__name__)

class Sidebar:
    """Application sidebar with navigation and settings."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def display(self):
        """Display the sidebar with navigation and settings."""
        try:
            with st.sidebar:
                # Header with app title and subtitle
                st.markdown("""
                <div class="sidebar-header">
                    <div class="gradient-text">MNIST App</div>
                    <div class="sidebar-subheader">Digit Classification AI</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Navigation buttons with container
                st.markdown('<div class="nav-buttons-container">', unsafe_allow_html=True)
                
                # Home button
                if st.button("🏠 Home", key="nav_home_btn", use_container_width=True, type="primary"):
                    NavigationState.set_active_view("home")
                    st.rerun()
                
                # Draw button
                if st.button("✏️ Draw & Classify", key="nav_draw_btn", use_container_width=True, type="primary"):
                    NavigationState.set_active_view("draw")
                    st.rerun()
                    
                # History button
                if st.button("📊 View History", key="nav_history_btn", use_container_width=True, type="primary"):
                    NavigationState.set_active_view("history")
                    st.rerun()
                
                # Settings button
                if st.button("⚙️ Settings", key="nav_settings_btn", use_container_width=True, type="primary"):
                    NavigationState.set_active_view("settings")
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add a visual divider
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Theme toggle section with centering container

                theme_manager = ThemeManager()
                bb8_toggle = BB8Toggle(theme_manager, on_change=None)
                bb8_toggle.render()
                
                # Add another visual divider
                
                # Footer info in sidebar
                st.markdown("""
                <div class="sidebar-footer">
                    <p>Version 1.0.0</p>
                    <p>© 2025 MNIST Classifier</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Error displaying sidebar: {str(e)}", exc_info=True)
            st.sidebar.error("Error in navigation")
            
            # Emergency navigation in case of errors
            st.sidebar.write("Emergency Navigation")
            if st.sidebar.button("🏠 Home"):
                NavigationState.navigate_to("home")
                st.rerun()