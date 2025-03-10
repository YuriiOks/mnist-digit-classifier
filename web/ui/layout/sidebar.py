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

logger = logging.getLogger(__name__)

class Sidebar:
    """Application sidebar with navigation and settings."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def display(self):
        """Display the sidebar with navigation and settings."""
        try:
            with st.sidebar:
                # Fancy header with gradient text
                st.markdown("""
                <style>
                .sidebar-header {
                    text-align: center;
                    margin-bottom: 20px;
                }
                .gradient-text {
                    background: linear-gradient(90deg, #4361ee, #4cc9f0);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: 700;
                    font-size: 2.2rem;
                    line-height: 1.2;
                    margin-bottom: 0.2rem;
                }
                .sidebar-subheader {
                    opacity: 0.8;
                    font-size: 0.9rem;
                    margin-bottom: 15px;
                    text-align: center;
                }
                
                /* Custom button styling */
                div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button {
                    width: 100% !important;
                    border-radius: 8px !important;
                    margin-bottom: 8px !important;
                    font-weight: 500 !important;
                    border: none !important;
                    padding: 10px 15px !important;
                    transition: all 0.3s !important;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
                }
                
                div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
                }
                
                .divider {
                    margin: 20px 0;
                    border-top: 1px solid rgba(128, 128, 128, 0.2);
                }
                </style>
                
                <div class="sidebar-header">
                    <div class="gradient-text">MNIST App</div>
                    <div class="sidebar-subheader">Digit Classification AI</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Navigation buttons - consistent styling
                st.markdown("### Navigation")
                
                # Home button
                if st.button("🏠 Home", key="nav_home_btn", use_container_width=True, type="primary"):
                    NavigationState.navigate_to("home")
                    st.rerun()
                
                # Draw button
                if st.button("✏️ Draw & Classify", key="nav_draw_btn", use_container_width=True, type="primary"):
                    NavigationState.navigate_to("draw")
                    st.rerun()
                
                # History button
                if st.button("📊 View History", key="nav_history_btn", use_container_width=True, type="primary"):
                    NavigationState.navigate_to("history")
                    st.rerun()
                
                # Settings button
                if st.button("⚙️ Settings", key="nav_settings_btn", use_container_width=True, type="primary"):
                    NavigationState.navigate_to("settings")
                    st.rerun()
                
                # Add a visual divider
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Theme toggle
                st.markdown("### Appearance")
                is_dark = st.toggle("Dark Mode", 
                                  ThemeState.is_dark_mode(),
                                  key="theme_toggle_sidebar")
                
                # Handle theme change
                if is_dark != ThemeState.is_dark_mode():
                    ThemeState.set_theme_mode("dark" if is_dark else "light")
                    ThemeManager.toggle_theme()
                    st.rerun()
                
                # Add another visual divider
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Footer info in sidebar
                st.markdown("""
                <div style="text-align: center; font-size: 0.8rem; opacity: 0.7; margin-top: 30px;">
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