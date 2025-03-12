# MNIST Digit Classifier
# Copyright (c) 2025
# File: app.py
# Description: Main application entry point
# Created: 2024-05-01

import streamlit as st
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("mnist_app.log"), 
                             logging.StreamHandler()])

logger = logging.getLogger(__name__)

# Add the current directory to the path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    logger.info(f"Added {current_dir} to Python path")

# First, configure page BEFORE any other Streamlit commands
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import core modules
from core.app_state.session_state import SessionState
from core.app_state.theme_state import ThemeState
from core.app_state.navigation_state import NavigationState
from core.registry.view_registry import ViewRegistry
from ui.layout.header import Header
from ui.layout.footer import Footer
from ui.layout.sidebar import Sidebar
from ui.theme.theme_manager import ThemeManager
from utils.css.font_loader import load_app_fonts
from utils.ui.layout import fix_layout_issues
from utils.css.css_loader import load_theme_css
from utils.css.components_css import apply_component_styles


def initialize_app():
    """Initialize application state and components."""
    logger.info("Initializing application...")
    
    # Initialize session state
    SessionState.initialize()
    
    # Initialize theme state
    ThemeState.initialize()
    
    # Initialize navigation state
    NavigationState.initialize()
    
    # Register available views
    ViewRegistry.register_default_views()
    
    logger.info("Application initialization complete")


def main():
    """Main application entry point."""
    try:
        # Initialize the application
        initialize_app()
        
        # Apply layout fixes for Streamlit UI
        fix_layout_issues()
        
        # Apply component styles
        apply_component_styles()
        
        # Load fonts
        load_app_fonts()
        
        # Initialize and apply theme
        theme_manager = ThemeManager()
        current_theme = ThemeState.get_current_theme()
        theme_manager.apply_theme(current_theme)
        
        # Load theme CSS
        load_theme_css(current_theme)
        
        # Display header
        header = Header("MNIST Digit Classifier")
        header.display()
        
        # Display sidebar navigation using our Sidebar class
        sidebar = Sidebar()
        sidebar.display()
        
        # Get active view from navigation state
        active_view_id = NavigationState.get_active_view()
        active_view = ViewRegistry.get_view(active_view_id)
        
        # Display the active view
        if active_view:
            active_view.display()
        else:
            st.error(f"View '{active_view_id}' not found")
            fallback_view = ViewRegistry.get_view("home")
            if fallback_view:
                fallback_view.display()
        
        # Display footer
        footer = Footer()
        footer.display()

        # # Register the callback function in Streamlit
        # st.components.v1.html(
        #     """
        #     <script>
        #     // Define the function in the Streamlit context
        #     window.callToggleTheme = function(theme) {
        #         // Use Streamlit's setComponentValue for more reliability
        #         if (window.parent && window.parent.streamlit) {
        #             const data = {
        #                 theme: theme,
        #                 isInitialCall: false
        #             };
        #             window.parent.streamlit.setComponentValue(data);
        #         }
        #     }
        #     </script>
        #     """,
        #     height=0,  # Make it invisible
        # )

        # Define a callback handler for theme toggle
        def handle_theme_toggle(theme):
            if theme and theme != st.session_state.get("current_theme"):
                ThemeManager.toggle_theme(theme)
                st.rerun()  # Force a rerun to apply the theme change

        # Register the component callback
        # components.declare_component(
        #     "theme_toggle_callback",
        #     handle_theme_toggle
        # )

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
