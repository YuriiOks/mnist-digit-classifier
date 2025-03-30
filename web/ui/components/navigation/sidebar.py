# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/navigation/sidebar.py
# Description: Sidebar navigation component
# Created: 2024-05-01

import streamlit as st
import logging
import uuid
from typing import List, Dict, Any, Optional, Union

from ui.components.base.component import Component
from core.app_state.navigation_state import NavigationState
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class Sidebar(Component):
    """Sidebar navigation component for the application."""

    def __init__(self, title: str = "Navigation"):
        """Initialize the sidebar component.

        Args:
            title: Title to display at the top of the sidebar
        """
        super().__init__(component_type="layout", component_name="sidebar")
        self.title = title
        self.logger = logging.getLogger(__name__)

    def display(self) -> None:
        """Display the sidebar component."""
        try:
            # Initialize theme variables with defaults
            theme_icon = "🌙"
            theme_name = "Dark Mode"

            # Get current theme and set appropriate icon/name
            try:
                current_theme = ThemeManager.get_theme()
                if current_theme == "dark":
                    theme_icon = "☀️"
                    theme_name = "Light Mode"
                else:
                    theme_icon = "🌙"
                    theme_name = "Dark Mode"
            except Exception as e:
                self.logger.error(f"Error getting theme: {str(e)}")

            # Display the sidebar title
            st.sidebar.title(self.title)

            # Add a separator
            st.sidebar.markdown("---")

            # Display navigation buttons
            st.sidebar.markdown("### Menu")

            # Get all registered views and active view
            routes = NavigationState.get_routes()
            active_view = NavigationState.get_active_view()

            # Render each navigation item
            for route in routes:
                # Create a button for each route with its icon
                if st.sidebar.button(
                    f"{route['icon']} {route['label']}",
                    key=f"nav_{route['id']}",
                    # Highlight active view
                    type=("primary" if active_view == route["id"] else "secondary"),
                    use_container_width=True,
                ):
                    try:
                        # Only set the active_view directly in session state, don't modify nav_history directly
                        st.session_state["active_view"] = route["id"]

                        # Use NavigationState to track history properly
                        NavigationState.set_active_view(route["id"])

                        # Debug message
                        self.logger.info(f"Navigating to {route['id']}")

                        # Force a rerun of the app to show the new view
                        st.rerun()
                    except Exception as e:
                        self.logger.error(
                            f"Error navigating to {route['id']}: {str(e)}"
                        )

            # Add a separator
            st.sidebar.markdown("---")

            # Single theme toggle button
            if st.sidebar.button(
                f"{theme_icon} {theme_name}",
                key="theme_toggle",
                use_container_width=True,
            ):
                try:
                    ThemeManager.toggle_theme()
                    st.rerun()
                except Exception as e:
                    self.logger.error(f"Error toggling theme: {str(e)}")

            self.logger.debug("Sidebar navigation displayed successfully")
        except Exception as e:
            self.logger.error(f"Error displaying sidebar: {str(e)}", exc_info=True)
            st.sidebar.error("Error loading navigation")

            # Fallback navigation in case of error
            st.sidebar.markdown("### Emergency Navigation")
            if st.sidebar.button("🏠 Home", key="emergency_home"):
                try:
                    st.session_state["active_view"] = "home"
                    st.rerun()
                except Exception as e:
                    self.logger.error(f"Error in emergency navigation: {str(e)}")
