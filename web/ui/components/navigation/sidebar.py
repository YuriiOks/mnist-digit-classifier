# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/navigation/sidebar.py
# Description: Navigation sidebar component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Callable

from ui.components.base.component import Component
from core.app_state.navigation_state import NavigationState
from core.registry.view_registry import ViewRegistry
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)

class Sidebar(Component):
    """Sidebar navigation component for the application."""
    
    def __init__(
        self,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize the sidebar component."""
        logger.debug("Initializing Sidebar")
        super().__init__(
            component_type="navigation",
            component_name="sidebar",
            id=id,
            classes=classes,
            attributes=attributes
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def display(self) -> None:
        """Display the sidebar component."""
        self.logger.debug("Displaying sidebar component")
        
        try:
            # Add custom CSS to ensure buttons are styled consistently
            st.sidebar.markdown("""
            <style>
            /* Consistent styling for all sidebar buttons */
            [data-testid="stSidebar"] button[kind="secondary"] {
                background: linear-gradient(90deg, var(--color-primary-light, #6366F1), var(--color-primary, #4F46E5)) !important;
                color: white !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 0.5rem 1rem !important;
                font-weight: 500 !important;
                font-family: var(--font-primary, 'Poppins'), sans-serif !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
                transition: all 0.3s ease !important;
                width: 100% !important;
                margin-bottom: 0.5rem !important;
                text-align: left !important;
                position: relative !important;
                overflow: hidden !important;
            }
            
            /* Hover effect for all sidebar buttons */
            [data-testid="stSidebar"] button[kind="secondary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
            }
            
            /* Active state for sidebar buttons */
            [data-testid="stSidebar"] button[kind="secondary"].active {
                background: linear-gradient(90deg, var(--color-primary, #4F46E5), var(--color-primary-dark, #3730A3)) !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            }
            
            /* Shine effect for sidebar buttons */
            [data-testid="stSidebar"] button[kind="secondary"]::after {
                content: '' !important;
                position: absolute !important;
                top: -50% !important;
                left: -50% !important;
                width: 200% !important;
                height: 200% !important;
                background: linear-gradient(
                    to right,
                    rgba(255, 255, 255, 0) 0%,
                    rgba(255, 255, 255, 0.3) 50%,
                    rgba(255, 255, 255, 0) 100%
                ) !important;
                transform: rotate(30deg) !important;
                opacity: 0 !important;
                transition: opacity 0.3s ease !important;
                pointer-events: none !important;
            }
            
            [data-testid="stSidebar"] button[kind="secondary"]:hover::after {
                opacity: 1 !important;
                animation: buttonShine 1.5s ease-in-out !important;
            }
            
            @keyframes buttonShine {
                0% {
                    transform: rotate(30deg) translate(-100%, -100%) !important;
                }
                100% {
                    transform: rotate(30deg) translate(100%, 100%) !important;
                }
            }
            
            /* Make icons stand out more */
            [data-testid="stSidebar"] button[kind="secondary"] div:first-child {
                display: inline-flex !important;
                margin-right: 0.5rem !important;
                font-size: 1.1rem !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Get active view
            active_view = NavigationState.get_active_view()
            self.logger.debug(f"Active view: {active_view}")
            
            # Display navigation items
            st.sidebar.markdown("### Navigation")
            
            # Get available views from registry
            navigation_items = NavigationState.get_routes()
            self.logger.debug(f"Navigation items: {navigation_items}")
            
            # Display each navigation item as a button
            for item in navigation_items:
                view_id = item.get("id", "")
                label = item.get("label", view_id.capitalize())
                icon = item.get("icon", "")
                
                # Add is_active class to active button using JavaScript
                active_class = "active" if view_id == active_view else ""
                
                # Make all buttons look consistent with the same styling
                if st.sidebar.button(f"{icon} {label}", key=f"nav_{view_id}", type="secondary"):
                    NavigationState.set_active_view(view_id)
                    st.rerun()
                
            # Apply active styling with JavaScript
            st.sidebar.markdown(f"""
            <script>
                // Mark the active button
                document.addEventListener('DOMContentLoaded', function() {{
                    // Find button for active view: {active_view}
                    const buttons = document.querySelectorAll('[data-testid="stSidebar"] button[kind="secondary"]');
                    for (const button of buttons) {{
                        if (button.innerText.includes("{navigation_items[next((i for i, item in enumerate(navigation_items) if item.get('id') == active_view), 0)].get('icon', '')}")) {{
                            button.classList.add('active');
                        }}
                    }}
                }});
            </script>
            """, unsafe_allow_html=True)
            
            # Simple separator
            st.sidebar.markdown('<hr style="margin: 1.5rem 0; opacity: 0.3;">', unsafe_allow_html=True)
            
            # Create a themed footer toggle at the bottom of the sidebar
            theme_name = "Dark Mode" if ThemeManager.get_theme() == "light" else "Light Mode"
            theme_icon = "🌙" if ThemeManager.get_theme() == "light" else "☀️"
            
            # Add spacer before the theme toggle button
            st.sidebar.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
            
            # Add custom CSS for the beautiful theme toggle button
            st.sidebar.markdown("""
            <style>
                /* Position button at the bottom of the sidebar, centered */
                .theme-toggle-container {
                    position: fixed;
                    bottom: 20px;
                    left: 50%; /* Center based on sidebar */
                    transform: translateX(-50%);
                    width: 80%;
                    max-width: 200px;
                    text-align: center;
                    z-index: 1000;
                }
                
                /* Special styling with distinct color scheme - using !important to override Streamlit styles */
                .theme-toggle-button {
                    /* Gold/amber gradient for light mode */
                    background: linear-gradient(135deg, #FF9500 0%, #FFCC00 100%) !important;
                    background-color: #FFCC00 !important; /* Fallback */
                    color: #222 !important;
                    border: none !important;
                    padding: 10px 16px !important;
                    border-radius: 20px !important;
                    font-size: 0.9rem !important;
                    font-weight: 600 !important;
                    cursor: pointer !important;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
                    transition: all 0.3s ease !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    width: 100% !important;
                    margin: 0 auto !important;
                }
                
                .theme-toggle-button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2) !important;
                }
                
                /* Dark mode version - cool teal/blue gradient */
                [data-theme="dark"] .theme-toggle-button {
                    background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
                    background-color: #00C9FF !important; /* Fallback */
                    color: #222 !important;
                }

                /* Add shimmer effect to the button */
                .theme-toggle-button::after {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: linear-gradient(
                        to right,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.4) 50%,
                        rgba(255, 255, 255, 0) 100%
                    ) !important;
                    transform: rotate(30deg);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    pointer-events: none;
                }
                
                .theme-toggle-button:hover::after {
                    opacity: 1;
                    animation: buttonShine 1.5s ease-in-out;
                }
                
                @keyframes buttonShine {
                    0% {
                        transform: rotate(30deg) translate(-100%, -100%);
                    }
                    100% {
                        transform: rotate(30deg) translate(100%, 100%);
                    }
                }

                /* Hide the original theme toggle button */
                div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:has(div:contains("🌙")),
                div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:has(div:contains("☀️")) {
                    display: none !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a properly styled button
            if st.sidebar.button(f"{theme_icon} {theme_name}", key="toggle_theme", type="secondary"):
                ThemeManager.toggle_theme()
                st.rerun()
            
            # Add enhanced background styling for the sidebar
            st.sidebar.markdown("""
            <style>
                /* Main sidebar container */
                section[data-testid="stSidebar"] {
                    background: linear-gradient(160deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 242, 255, 1) 100%);
                    border-right: 1px solid rgba(0, 0, 0, 0.08);
                    box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.05);
                }
                
                /* Sidebar content area */
                [data-testid="stSidebarContent"] {
                    background: linear-gradient(160deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 242, 255, 1) 100%);
                }
                
                /* Add subtle texture pattern */
                [data-testid="stSidebarUserContent"]::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    opacity: 0.05;
                    pointer-events: none;
                    background-image: 
                        radial-gradient(var(--color-primary, #4F46E5) 1px, transparent 1px),
                        radial-gradient(var(--color-secondary, #06B6D4) 1px, transparent 1px);
                    background-size: 20px 20px;
                    background-position: 0 0, 10px 10px;
                    z-index: 0;
                }
                
                /* Sidebar header */
                [data-testid="stSidebarHeader"] {
                    background: transparent !important;
                }
                
                /* All sidebar content */
                [data-testid="stSidebarUserContent"] > div {
                    background: transparent !important;
                    z-index: 1;
                    position: relative;
                }
                
                /* Dark mode adjustments */
                [data-theme="dark"] section[data-testid="stSidebar"],
                [data-theme="dark"] [data-testid="stSidebarContent"] {
                    background: linear-gradient(160deg, rgba(30, 30, 40, 0.98) 0%, rgba(15, 15, 25, 1) 100%);
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                /* Sidebar title area enhancement */
                .sidebar-title {
                    margin-bottom: 1.5rem !important;
                    position: relative;
                    z-index: 1;
                }
                
                /* Colorful top border */
                [data-testid="stSidebarUserContent"]::after {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, var(--color-primary, #4F46E5), var(--color-secondary, #06B6D4));
                    opacity: 0.8;
                    z-index: 100;
                }
                
                /* Hide default Streamlit header border */
                [data-testid="stSidebarHeader"] {
                    border-bottom: none !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            self.logger.debug("Sidebar navigation displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying sidebar: {str(e)}", exc_info=True)
            st.sidebar.error("Error loading navigation")
            
            # Fallback navigation
            st.sidebar.markdown("### Emergency Navigation")
            if st.sidebar.button("🏠 Home"):
                NavigationState.set_active_view("home")
                st.rerun()

            # Position theme toggle button with custom colors and centered placement
            st.sidebar.markdown("""
            <style>
                /* Position button at the bottom of the sidebar, centered */
                .theme-toggle-container {
                    position: fixed;
                    bottom: 20px;
                    left: 50%; /* Center based on sidebar */
                    transform: translateX(-50%);
                    width: 80%;
                    max-width: 200px;
                    text-align: center;
                    z-index: 1000;
                }
                
                /* Special styling with distinct color scheme - using !important to override Streamlit styles */
                .theme-toggle-button {
                    /* Gold/amber gradient for light mode */
                    background: linear-gradient(135deg, #FF9500 0%, #FFCC00 100%) !important;
                    background-color: #FFCC00 !important; /* Fallback */
                    color: #222 !important;
                    border: none !important;
                    padding: 10px 16px !important;
                    border-radius: 20px !important;
                    font-size: 0.9rem !important;
                    font-weight: 600 !important;
                    cursor: pointer !important;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
                    transition: all 0.3s ease !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    width: 100% !important;
                    margin: 0 auto !important;
                }
                
                .theme-toggle-button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2) !important;
                }
                
                /* Dark mode version - cool teal/blue gradient */
                [data-theme="dark"] .theme-toggle-button {
                    background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%) !important;
                    background-color: #00C9FF !important; /* Fallback */
                    color: #222 !important;
                }

                /* Add shimmer effect to the button */
                .theme-toggle-button::after {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: linear-gradient(
                        to right,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.4) 50%,
                        rgba(255, 255, 255, 0) 100%
                    ) !important;
                    transform: rotate(30deg);
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    pointer-events: none;
                }
                
                .theme-toggle-button:hover::after {
                    opacity: 1;
                    animation: buttonShine 1.5s ease-in-out;
                }
                
                @keyframes buttonShine {
                    0% {
                        transform: rotate(30deg) translate(-100%, -100%);
                    }
                    100% {
                        transform: rotate(30deg) translate(100%, 100%);
                    }
                }

                /* Hide the original theme toggle button */
                div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:has(div:contains("🌙")),
                div[data-testid="stSidebarUserContent"] button[data-testid="stBaseButton-secondary"]:has(div:contains("☀️")) {
                    display: none !important;
                }
            </style>

            <div class="theme-toggle-container">
                <button onclick="toggleTheme()" class="theme-toggle-button">
                    {theme_icon} {theme_name}
                </button>
            </div>

            <script>
                function toggleTheme() {
                    // Find and click the hidden toggle theme button
                    const buttons = document.querySelectorAll('button[data-testid="stBaseButton-secondary"]');
                    for (const button of buttons) {
                        if (button.innerText.includes("🌙") || button.innerText.includes("☀️")) {
                            button.click();
                            break;
                        }
                    }
                }
            </script>
            """.format(theme_icon=theme_icon, theme_name=theme_name), unsafe_allow_html=True)
            
            # Keep the original button for functionality, but it will be hidden by CSS
            if st.sidebar.button(f"{theme_icon} {theme_name}", key="toggle_theme", type="secondary"):
                ThemeManager.toggle_theme()
                st.rerun()
        except Exception as e:
            self.logger.error(f"Error displaying sidebar: {str(e)}", exc_info=True)
            st.sidebar.error("Error loading navigation")
            
            # Fallback navigation
            st.sidebar.markdown("### Emergency Navigation")
            if st.sidebar.button("🏠 Home"):
                NavigationState.set_active_view("home")
                st.rerun() 