# MNIST Digit Classifier
# Copyright (c) 2025
# File: app.py
# Description: Main application entry point with view-based architecture
# Created: 2025-03-17
# Updated: 2025-03-24

import streamlit as st
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mnist_app.log")
    ]
)

logger = logging.getLogger("mnist_app")

# Set environment variable for development mode
os.environ["MNIST_ENV"] = os.environ.get("MNIST_ENV", "development")

# Import database components first to ensure initialization before app state
from core.database import initialize_database
db = initialize_database()
logger.info("Database initialized successfully")

# Import core components
from core.app_state import initialize_app_state
from core.app_state.navigation_state import NavigationState
from utils.resource_manager import resource_manager
from ui.theme.theme_manager import theme_manager
from ui.layout.layout import Layout

# Import views (this is new)
from ui.views import views

def load_core_css():
    """Load core CSS files with improved error handling and fallbacks."""
    # Determine if we're in Docker or development
    in_docker = Path("/app").exists()
    logger.info(f"Running in Docker: {in_docker}")
    
    # Paths relative to assets/css with fallbacks
    css_files = [
        # Global styles
        "global/variables.css",
        "global/animations.css",
        "global/base.css", 
        "global/reset.css",

        # Cards
        "components/cards/cards.css",

        # Controls
        "components/controls/bb8-toggle.css",
        "components/controls/buttons.css",
        "components/controls/pagination.css",

        # Layout
        "components/layout/header.css",        
        "components/layout/footer.css",

        # Inputs
        "components/inputs/canvas.css",
        "components/inputs/file-upload.css",


        # Theme system
        "themes/theme-system.css",
        
        # View styles
        "views/view_styles.css",
    ]
    
    # Try to load each file
    loaded_css = ""
    loaded_count = 0
    failed_count = 0
    
    for css_file in css_files:
        css_content = resource_manager.load_css(css_file)
        if css_content:
            loaded_css += f"\n/* {css_file} */\n{css_content}"
            loaded_count += 1
        else:
            logger.warning(f"Could not load core CSS: {css_file}")
            failed_count += 1
    
    # Inject the combined CSS
    if loaded_css:
        resource_manager.inject_css(loaded_css)
        logger.info(f"Successfully loaded {loaded_count} CSS files, {failed_count} failed")
    else:
        logger.error("No core CSS was loaded - app may display incorrectly")

def main():
    """Main entry point for the MNIST Digit Classifier application."""
    st.set_page_config(
        page_title="MNIST Digit Classifier",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize application state
    initialize_app_state()
    
    # Load core CSS files
    load_core_css()
    
    # Create the layout manager with extensive logging
    logger.info("Creating Layout manager")
    layout = Layout(title="MNIST Digit Classifier")
    
    # Render sidebar directly through the Layout manager
    layout.render_sidebar()
    
    # Render header 
    layout.render_header()
    
    # Get current view and render its content
    current_view_name = NavigationState.get_active_view()
    logger.info(f"Current view: {current_view_name}")
    
    # Get the view instance from our views dictionary
    current_view = views.get(current_view_name)
    
    # Render the view if it exists
    if current_view:
        logger.info(f"Rendering view: {current_view.name}")
        current_view.display()
        logger.info(f"View rendered successfully: {current_view.name}")
    else:
        st.error(f"View '{current_view_name}' not found.")
        logger.error(f"View not found: {current_view_name}")
        # Fall back to home view
        logger.info("Falling back to home view")
        views["home"].display()

    # Render footer last
    layout.render_footer()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("The application encountered a critical error. Please check the logs for details.")
        st.code(str(e))