# MNIST Digit Classifier
# File: app.py
# Description: Main application entry point
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Optional, Dict, Any
import os
import sys
import pathlib

# Add the current directory to the path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Added {current_dir} to Python path")

# First, configure page BEFORE any other Streamlit commands
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global layout fixes
try:
    from utils.ui.layout import fix_layout_issues
    fix_layout_issues()  # Apply layout fixes globally
except ImportError as e:
    st.warning(f"Could not import layout fixes: {str(e)}")
    # Add basic fixes directly as a fallback
    st.markdown("""
    <style>
    /* Basic layout fixes */
    .app-header, .app-footer {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

# IMMEDIATELY INJECT FONTS - This is a direct approach to ensure fonts load
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap');

/* Define font variables */
:root {
    --font-primary: 'Poppins', sans-serif;
    --font-secondary: 'Nunito', sans-serif;
    --font-mono: 'Roboto Mono', monospace;
}

/* Apply fonts directly */
body, .stApp {
    font-family: var(--font-secondary) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-primary) !important;
}

code, pre {
    font-family: var(--font-mono) !important;
}

/* Basic theme variables for immediate styling */
body {
    --color-background: #f8f9fa;
    --color-card: #ffffff;
    --color-text: #333333;
    --color-text-light: #666666;
    --color-primary: #4F46E5;
    --color-primary-dark: #4338CA;
    --color-secondary: #06B6D4;
    --color-accent: #F59E0B;
    --color-border: #e0e0e0;
    --border-radius-md: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
}

/* Dark mode colors (will be applied via JS later) */
[data-theme="dark"] {
    --color-background: #1a1a1a;
    --color-card: #2a2a2a;
    --color-text: #e0e0e0;
    --color-text-light: #b0b0b0;
    --color-primary: #818CF8;
    --color-primary-dark: #6366F1;
    --color-secondary: #22D3EE;
    --color-accent: #FBBF24;
    --color-border: #444444;
}
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to stdout for Streamlit to capture
        logging.FileHandler("mnist_app.log")  # Also log to file
    ]
)

# Create a module-level logger
logger = logging.getLogger(__name__)

# SAFETY MECHANISM: Add a try-except around the ENTIRE app
try:
    # Create debug module directories if they don't exist
    def ensure_debug_modules():
        """Ensure debug module directories and __init__.py files exist."""
        debug_dir = os.path.join("ui", "views", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create __init__.py files if they don't exist
        for path in ["ui/views/__init__.py", "ui/views/debug/__init__.py"]:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write("# Auto-generated init file\n")

    # Call this early to ensure the directories exist
    ensure_debug_modules()
    
    # Import core components
    logger.info("Initializing application components")
    
    # Import module-level components
    from core.app_state.navigation_state import NavigationState
    from core.registry.view_registry import ViewRegistry
    from ui.theme.theme_manager import ThemeManager
    
    # Initialize app state
    from core.app_state import initialize_app_state
    initialize_app_state()
    
    # Initialize theme and load styles
    ThemeManager.initialize()
    ThemeManager.inject_card_styles()  # Explicitly load card styles
    
    # Register all views
    def register_application_views():
        """Register all application views with the view registry."""
        logger.info("Registering application views")
        
        try:
            # Import and register views
            from ui.views.home.home_view import HomeView
            from ui.views.drawing.drawing_view import DrawingView
            from ui.views.history.history_view import HistoryView
            from ui.views.settings.settings_view import SettingsView
            from ui.views.debug.debug_view import DebugView
            from ui.views.debug.css_debugger import CSSDebugView
            from ui.views.debug.font_tester import FontTesterView
            
            # Create and register views
            ViewRegistry.register_view(HomeView())
            ViewRegistry.register_view(DrawingView())
            ViewRegistry.register_view(HistoryView())
            ViewRegistry.register_view(SettingsView())
            ViewRegistry.register_view(DebugView())
            ViewRegistry.register_view(CSSDebugView())
            ViewRegistry.register_view(FontTesterView())
            
            logger.info(f"Registered {len(ViewRegistry.get_all_views())} views")
        except Exception as e:
            logger.error(f"Error registering views: {str(e)}", exc_info=True)
            raise
    
    # Register views
    register_application_views()
    
    # Load CSS - if we have the css_loader module
    try:
        from utils.css.css_loader import load_all_css
        load_all_css()
    except ImportError:
        logger.warning("CSS loader not available")
    
    # Simple emergency UI function
    def show_emergency_ui():
        """Show a simple UI when the main UI fails to load."""
        st.title("MNIST Digit Classifier - Emergency Mode")
        st.warning("The application is currently in emergency mode due to an error.")
        
        st.write("### Debug Information")
        st.code(f"Python path: {sys.path}")
        st.code(f"Current directory: {os.getcwd()}")
        
        st.write("### Last Error")
        if 'last_error' in st.session_state:
            st.error(st.session_state['last_error'])
        
        st.write("### Manual Navigation")
        if st.button("Home View"):
            NavigationState.set_active_view("home")
            st.rerun()
            
        if st.button("Debug View"):
            NavigationState.set_active_view("debug")
            st.rerun()
            
        if st.button("CSS Debug View"):
            NavigationState.set_active_view("css_debug")
            st.rerun()
        
        st.write("### Application State")
        st.json(dict(st.session_state))

    # Main application rendering
    try:
        # Display the header
        from ui.layout.header import Header
        header = Header(title="MNIST Digit Classifier")
        header.display()
        
        # Create the layout - FIXED: remove the column approach
        from ui.components.navigation.sidebar import Sidebar
        sidebar = Sidebar()
        sidebar.display()
        
        # Render the active view in the main area
        try:
            active_view_id = NavigationState.get_active_view()
            logger.info(f"Rendering active view: {active_view_id}")
            
            # Get and render the active view
            active_view = ViewRegistry.get_view(active_view_id)
            if active_view:
                active_view.render()
            else:
                st.error(f"View not found: {active_view_id}")
                logger.error(f"View not found: {active_view_id}")
                # Try to fallback to home view
                home_view = ViewRegistry.get_view("home")
                if home_view:
                    home_view.render()
        except Exception as e:
            logger.error(f"Error rendering view: {str(e)}", exc_info=True)
            st.error(f"Error rendering view: {str(e)}")
            st.session_state['last_error'] = f"View rendering error: {str(e)}"
        
        # Display the footer
        from ui.layout.footer import Footer
        footer = Footer()
        footer.display()
        
    except Exception as e:
        logger.error(f"Error in main UI: {e}", exc_info=True)
        st.session_state['last_error'] = str(e)
        show_emergency_ui()

except Exception as e:
    st.error(f"Critical application error: {str(e)}")
    st.write("Please check the logs or contact support.")
    logger.critical(f"Critical application error: {str(e)}", exc_info=True)
