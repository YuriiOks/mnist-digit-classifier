# MNIST Digit Classifier
# Copyright (c) 2025
# File: app.py
# Description: Main application entry point
# Created: 2025-03-16

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

# Import core components
from core.app_state import initialize_app_state
from core.app_state.navigation_state import NavigationState
from utils.resource_manager import resource_manager
from ui.theme.theme_manager import theme_manager
from ui.layout.layout_components import Layout
from ui.components.cards.card import Card, FeatureCard, WelcomeCard


def load_core_css():
    """Load core CSS files."""
    # Direct CSS paths that should match the project structure
    css_files = [
        "global/variables.css",
        "global/reset.css",
        "global/base.css", 
        "global/animations.css",
        "themes/theme-system.css"
    ]
    
    # Try to load each file
    loaded_css = ""
    for css_file in css_files:
        css_content = resource_manager.load_css(css_file)
        if css_content:
            loaded_css += f"\n/* {css_file} */\n{css_content}"
        else:
            logger.warning(f"Could not load core CSS: {css_file}")
    
    # Inject the combined CSS
    if loaded_css:
        resource_manager.inject_css(loaded_css)
    else:
        logger.warning("No core CSS was loaded")


def load_view_css(view_name):
    """
    Load view-specific CSS.
    
    Args:
        view_name: Name of the view to load CSS for.
    """
    resource_manager.load_and_inject_css([
        f"views/{view_name}.css",
        "views/view_styles.css"
    ])


def render_home_view():
    """Render the home view."""
    load_view_css("home")
    
    # Load welcome card data from JSON
    welcome_data = resource_manager.load_json_resource("home/welcome_card.json")
    if welcome_data:
        welcome_card = WelcomeCard(
            title=welcome_data.get("title", "MNIST Digit Classifier"),
            content=welcome_data.get("content", "Welcome to the MNIST Digit Classifier."),
            icon=welcome_data.get("icon", "👋")
        )
        welcome_card.display()
    else:
        # Fallback if data not found
        welcome_card = WelcomeCard(
            title="MNIST Digit Classifier",
            content="<b>Welcome to the MNIST Digit Classifier!</b><p>This application allows you to draw digits and have them recognized by a machine learning model.</p>",
            icon="👋"
        )
        welcome_card.display()
    
    # Display "How It Works" section
    st.markdown("<h2 id='how-it-works'>How It Works</h2>", unsafe_allow_html=True)
    
    # Load feature cards data from JSON
    feature_cards = resource_manager.load_json_resource("home/feature_cards.json")
    if feature_cards:
        # Create columns for feature cards
        cols = st.columns(len(feature_cards))
        
        # Display feature cards
        for i, card_data in enumerate(feature_cards):
            with cols[i]:
                feature_card = FeatureCard(
                    title=card_data.get("title", f"Feature {i+1}"),
                    content=card_data.get("content", ""),
                    icon=card_data.get("icon", "")
                )
                feature_card.display()
    else:
        # Fallback with basic feature cards
        cols = st.columns(3)
        
        with cols[0]:
            FeatureCard(
                title="1. Draw a Digit",
                content="<p>Navigate to the Drawing Canvas and use your mouse or touch to draw any digit from 0-9.</p>",
                icon="✏️"
            ).display()
        
        with cols[1]:
            FeatureCard(
                title="2. Classify the Digit",
                content="<p>Once you've drawn a digit, click the \"Classify\" button to have the model predict what number you've drawn.</p>",
                icon="🔍"
            ).display()
        
        with cols[2]:
            FeatureCard(
                title="3. View History",
                content="<p>All your predictions are saved to your session history.</p>",
                icon="📊"
            ).display()
    
    # CTA section
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="cta-section">', unsafe_allow_html=True)
    
    cta_col1, cta_col2 = st.columns([2, 3])
    
    with cta_col1:
        st.markdown("<h2 id='ready-to-try-it-out' style='margin: 0;'>Ready to try it out?</h2>", unsafe_allow_html=True)
    
    with cta_col2:
        if st.button("Start Drawing", type="primary"):
            NavigationState.set_active_view("draw")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_draw_view():
    """Render the draw view (placeholder)."""
    load_view_css("draw")
    
    st.write("### Draw a Digit")
    st.write("This is a placeholder for the drawing canvas. In a full implementation, this would include the drawing canvas and prediction UI.")
    
    # Basic canvas placeholder
    st.markdown("""
    <div style="
        width: 280px;
        height: 280px;
        border: 2px dashed #ccc;
        border-radius: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    ">
        <p style="color: #888888; font-size: 1.2em;">
            Drawing Canvas
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.button("Clear", type="secondary")
    
    with col2:
        if st.button("Predict", type="primary"):
            st.success("This is a simulated prediction: The digit is 5")


def render_history_view():
    """Render the history view (placeholder)."""
    load_view_css("history")
    
    st.write("### Prediction History")
    st.write("This is a placeholder for the prediction history view. In a full implementation, this would show a history of predictions and analytics.")
    
    # Simple table placeholder
    data = {
        "Time": ["2025-03-16 10:30", "2025-03-16 10:35", "2025-03-16 10:40"],
        "Digit": [5, 3, 7],
        "Confidence": ["95%", "87%", "92%"],
    }
    
    st.table(data)


def render_settings_view():
    """Render the settings view (placeholder)."""
    load_view_css("settings")
    
    st.write("### Settings")
    st.write("This is a placeholder for the settings view. In a full implementation, this would allow customizing app settings.")
    
    # Theme settings
    st.subheader("Appearance")
    theme_mode = theme_manager.get_current_theme()
    
    st.write(f"Current theme: **{theme_mode.capitalize()}**")
    if st.button(f"Switch to {'Light' if theme_mode == 'dark' else 'Dark'} Mode"):
        theme_manager.toggle_theme()
        st.rerun()
    
    # Other settings placeholders
    st.subheader("Canvas Settings")
    st.slider("Line thickness", 1, 10, 3)
    
    st.subheader("Application Settings")
    st.checkbox("Save prediction history", value=True)
    st.checkbox("Show tooltips", value=True)


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
    
    # Set up layout with header and footer
    layout = Layout(title="MNIST Digit Classifier")
    layout.render()
    
    # Determine which view to render based on navigation state
    current_view = NavigationState.get_active_view()
    
    # Render the appropriate view
    if current_view == "home":
        render_home_view()
    elif current_view == "draw":
        render_draw_view()
    elif current_view == "history":
        render_history_view()
    elif current_view == "settings":
        render_settings_view()
    else:
        # Fallback to home view
        render_home_view()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("The application encountered a critical error. Please check the logs for details.")
        st.code(str(e))

