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

import streamlit as st
from st_click_detector import click_detector

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
        
        # Theme system
        "themes/theme-system.css",
        
        # View styles
        "views/view_styles.css",
        "views/home.css",
        "views/draw.css",
        "views/history.css",
        "views/settings.css",
        
        # Components - use singular and plural variants
        "components/cards/card.css",
        "components/cards/cards.css",
        "components/controls/button.css", 
        "components/controls/buttons.css",
        "components/controls/bb8-toggle.css",  # Correct name
        
        # Other components
        "components/layout/footer.css",
        "components/layout/header.css",
        "components/layout/sidebar.css",
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
            # Try alternate names
            if "button.css" in css_file:
                alt_file = css_file.replace("button.css", "buttons.css")
                css_content = resource_manager.load_css(alt_file)
            elif "buttons.css" in css_file:
                alt_file = css_file.replace("buttons.css", "button.css")
                css_content = resource_manager.load_css(alt_file)
            elif "card.css" in css_file:
                alt_file = css_file.replace("card.css", "cards.css")
                css_content = resource_manager.load_css(alt_file)
            elif "cards.css" in css_file:
                alt_file = css_file.replace("cards.css", "card.css")
                css_content = resource_manager.load_css(alt_file)
                
            if css_content:
                loaded_css += f"\n/* {alt_file} (alt) */\n{css_content}"
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


    welcome_card = WelcomeCard(
        title=welcome_data.get("title", "MNIST Digit Classifier"),
        content=welcome_data.get("content", "Welcome to the MNIST Digit Classifier."),
        icon=welcome_data.get("icon", "👋")
    )
    welcome_card.display()
    
    # Display "How It Works" section
    st.markdown("<h2 id='how-it-works'>How It Works</h2>", unsafe_allow_html=True)
    
    # Load feature cards data from JSON
    feature_cards = resource_manager.load_json_resource("home/feature_cards.json")

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
    
    # CTA section
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="cta-section">', unsafe_allow_html=True)
    
    cta_col1, cta_col2 = st.columns([2, 3])
    
    with cta_col1:
        st.markdown("<h2 id='ready-to-try-it-out' style='margin: 0;'>Ready to try it out?</h2>", unsafe_allow_html=True)
    
    with cta_col2:
        ## Add custom CSS for the button

        if st.button("Start Drawing", type="primary"):
            NavigationState.set_active_view("draw")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_draw_view():
    """
    Render the draw view with multiple input options and prediction feedback.
    """
    import time
    import random
    
    load_view_css("draw")
    
    st.markdown("<h2>Digit Recognition</h2>", unsafe_allow_html=True)
    st.markdown("<p>Choose a method to input a digit for recognition.</p>", unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "draw"
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    if "prediction_correct" not in st.session_state:
        st.session_state.prediction_correct = None
    if "show_correction" not in st.session_state:
        st.session_state.show_correction = False
    if "predicted_digit" not in st.session_state:
        st.session_state.predicted_digit = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = "canvas_initial"
    if "file_upload_key" not in st.session_state:
        st.session_state.file_upload_key = "file_uploader_initial"
    if "url_input_key" not in st.session_state:
        st.session_state.url_input_key = "url_input_initial"
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    
    # Tab selection using buttons instead of st.tabs to keep content visible
    tab_cols = st.columns(3)
    with tab_cols[0]:
        if st.button("Draw Digit", 
                    key="tab_draw", 
                    type="primary" if st.session_state.active_tab == "draw" else "secondary",
                    use_container_width=True):
            st.session_state.active_tab = "draw"
            st.rerun()
    
    with tab_cols[1]:
        if st.button("Upload Image", 
                    key="tab_upload", 
                    type="primary" if st.session_state.active_tab == "upload" else "secondary",
                    use_container_width=True):
            st.session_state.active_tab = "upload"
            st.rerun()
    
    with tab_cols[2]:
        if st.button("Enter URL", 
                    key="tab_url", 
                    type="primary" if st.session_state.active_tab == "url" else "secondary",
                    use_container_width=True):
            st.session_state.active_tab = "url"
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Input container based on active tab
    if st.session_state.active_tab == "draw":
        st.markdown("<h3>Draw a Digit</h3>", unsafe_allow_html=True)
        st.markdown("<p>Use your mouse or touch to draw a digit from 0-9.</p>", unsafe_allow_html=True)
        
        # Add canvas for drawing
        try:
            from streamlit_drawable_canvas import st_canvas
            
            # Canvas configuration
            stroke_width = st.slider("Brush Width", 10, 25, 15)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=st.session_state.canvas_key,
                display_toolbar=True,
            )
        except ImportError:
            st.error("The streamlit-drawable-canvas package is not installed. Please install it with: pip install streamlit-drawable-canvas")
            
            # Show placeholder canvas
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
                    Drawing Canvas (Requires streamlit-drawable-canvas)
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.active_tab == "upload":
        st.markdown("<h3>Upload an Image</h3>", unsafe_allow_html=True)
        st.markdown("<p>Upload an image of a handwritten digit (PNG, JPG, JPEG).</p>", unsafe_allow_html=True)
        
        # File uploader - use the key from session state
        uploaded_file = st.file_uploader("Upload digit image", 
                                        type=["png", "jpg", "jpeg"], 
                                        key=st.session_state.file_upload_key)
        
        # Preview uploaded image
        if uploaded_file is not None:
            try:
                from PIL import Image
                import io
                
                # Read and display the image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=280)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    elif st.session_state.active_tab == "url":
        st.markdown("<h3>Enter Image URL</h3>", unsafe_allow_html=True)
        st.markdown("<p>Provide a URL to an image of a handwritten digit.</p>", unsafe_allow_html=True)
        
        # URL input - use the key from session state
        url = st.text_input("Image URL", 
                           key=st.session_state.url_input_key, 
                           placeholder="https://example.com/digit.jpg")
        
        # Load image if URL is provided
        if url:
            try:
                import requests
                from PIL import Image
                import io
                
                # Show loading indicator
                with st.spinner("Loading image from URL..."):
                    # Fetch image from URL
                    response = requests.get(url, timeout=5)
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        # Check if the content type is an image
                        content_type = response.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            st.error("The URL doesn't point to an image.")
                        else:
                            # Process and display the image
                            image = Image.open(io.BytesIO(response.content))
                            st.image(image, caption="Image from URL", width=280)
                    else:
                        st.error(f"Failed to load image. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")
    
    # Action buttons - always visible regardless of active tab
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Add buttons in a row
    button_cols = st.columns(2)
    
    # Clear button - resets everything by changing widget keys
    with button_cols[0]:
        if st.button("Clear All", key="clear_all", type="secondary", use_container_width=True):
            # Reset all prediction-related state
            st.session_state.prediction_made = False
            st.session_state.prediction_correct = None
            st.session_state.show_correction = False
            st.session_state.predicted_digit = None
            st.session_state.confidence = None
            
            # Generate new keys for all input widgets to effectively reset them
            timestamp = int(time.time() * 1000)
            st.session_state.canvas_key = f"canvas_{timestamp}"
            st.session_state.file_upload_key = f"file_uploader_{timestamp}"
            st.session_state.url_input_key = f"url_input_{timestamp}"
            st.session_state.reset_counter += 1
            
            # Trigger a rerun to apply the changes
            st.rerun()
    
    # Predict button
    with button_cols[1]:
        if st.button("Predict", key="predict", type="primary", use_container_width=True):
            # Simulated prediction
            st.session_state.predicted_digit = random.randint(0, 9)
            st.session_state.confidence = random.uniform(0.7, 0.99)
            st.session_state.prediction_made = True
            st.session_state.show_correction = False
            st.session_state.prediction_correct = None
    
    # ------ PREDICTION RESULT ------
    if st.session_state.prediction_made:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
        
        # Create two columns for the prediction display
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display the predicted digit prominently
            st.markdown(f"""
            <div style="text-align: center; font-size: 8rem; font-weight: bold; color: var(--color-primary);">
                {st.session_state.predicted_digit}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Display confidence information
            st.markdown("<h4>Confidence</h4>", unsafe_allow_html=True)
            
            # Format confidence as percentage
            confidence_pct = f"{st.session_state.confidence * 100:.1f}%"
            
            # Progress bar for confidence
            st.progress(st.session_state.confidence)
            st.markdown(f"<p>The model is {confidence_pct} confident in this prediction.</p>", unsafe_allow_html=True)
            
            # Feedback options
            st.markdown("<h4>Is this correct?</h4>", unsafe_allow_html=True)
            
            # Thumbs up/down buttons
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if st.button("👍 Yes", key=f"thumbs_up_{st.session_state.reset_counter}", use_container_width=True):
                    st.session_state.prediction_correct = True
                    st.session_state.show_correction = False
                    st.success("Thank you for your feedback!")
            
            with feedback_col2:
                if st.button("👎 No", key=f"thumbs_down_{st.session_state.reset_counter}", use_container_width=True):
                    st.session_state.prediction_correct = False
                    st.session_state.show_correction = True
        
        # Show correction input if thumbs down was clicked
        if st.session_state.show_correction:
            st.markdown("<h4>What's the correct digit?</h4>", unsafe_allow_html=True)
            
            # Create a row of digit buttons
            digit_cols = st.columns(10)
            for i in range(10):
                with digit_cols[i]:
                    if st.button(str(i), key=f"digit_{i}_{st.session_state.reset_counter}"):
                        # In a real app, you would save this correction to a database
                        st.session_state.corrected_digit = i
                        st.success(f"Thank you! Recorded the correct digit as {i}.")
                        st.session_state.show_correction = False

def find_files(base_dir, extension):
    """Find all files with given extension."""
    results = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                results.append(rel_path)
    return sorted(results)

def render_settings_view():
    """
    Render the settings view with various configuration options.
    
    Features:
    - Theme settings (light/dark mode, accent colors)
    - Canvas settings (size, stroke width, colors)
    - Prediction settings (confidence threshold, history size)
    - Application preferences
    """
    from core.app_state.settings_state import SettingsState
    from ui.theme.theme_manager import theme_manager
    
    load_view_css("settings")
    
    st.markdown("<h2>Settings</h2>", unsafe_allow_html=True)
    st.markdown("<p>Configure your preferences for the MNIST Digit Classifier.</p>", unsafe_allow_html=True)
    
    # Initialize the active settings tab if not already set
    if "active_settings_tab" not in st.session_state:
        st.session_state.active_settings_tab = "theme"
    
    # Create button-style tabs to match the draw view
    tab_cols = st.columns(4)
    
    with tab_cols[0]:
        if st.button("Theme Settings", 
                  key="tab_theme", 
                  type="primary" if st.session_state.active_settings_tab == "theme" else "secondary",
                  use_container_width=True):
            st.session_state.active_settings_tab = "theme"
            st.rerun()
    
    with tab_cols[1]:
        if st.button("Canvas Settings", 
                  key="tab_canvas", 
                  type="primary" if st.session_state.active_settings_tab == "canvas" else "secondary",
                  use_container_width=True):
            st.session_state.active_settings_tab = "canvas"
            st.rerun()
    
    with tab_cols[2]:
        if st.button("Prediction Settings", 
                  key="tab_prediction", 
                  type="primary" if st.session_state.active_settings_tab == "prediction" else "secondary",
                  use_container_width=True):
            st.session_state.active_settings_tab = "prediction"
            st.rerun()
    
    with tab_cols[3]:
        if st.button("App Settings", 
                  key="tab_app", 
                  type="primary" if st.session_state.active_settings_tab == "app" else "secondary",
                  use_container_width=True):
            st.session_state.active_settings_tab = "app"
            st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ===== THEME SETTINGS =====
    if st.session_state.active_settings_tab == "theme":
        st.markdown("<h3>Theme Settings</h3>", unsafe_allow_html=True)
        
        # Theme Mode
        st.markdown("<h4>Theme Mode</h4>", unsafe_allow_html=True)
        
        # Get current theme
        current_theme = theme_manager.get_current_theme()
        
        # Create theme selector with visual preview
        col1, col2 = st.columns(2)
        
        with col1:
            light_selected = current_theme == "light"
            if st.button(
                "Light Theme", 
                key="light_theme_btn",
                type="primary" if light_selected else "secondary",
                use_container_width=True
            ):
                # Apply the theme and save to settings
                theme_manager.apply_theme("light")
                SettingsState.set_setting("theme", "mode", "light")
                st.rerun()
                
            st.markdown("""
            <div style="
                border-radius: 8px;
                border: 1px solid #dee2e6;
                padding: 1rem;
                margin-top: 0.5rem;
                background-color: #f8f9fa;
                color: #212529;
                text-align: center;
            ">
                <div style="display: flex; justify-content: center; gap: 0.5rem;">
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #4361ee;"></div>
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #4cc9f0;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            dark_selected = current_theme == "dark"
            if st.button(
                "Dark Theme", 
                key="dark_theme_btn",
                type="primary" if dark_selected else "secondary",
                use_container_width=True
            ):
                # Apply the theme and save to settings
                theme_manager.apply_theme("dark")
                SettingsState.set_setting("theme", "mode", "dark")
                st.rerun()
                
            st.markdown("""
            <div style="
                border-radius: 8px;
                border: 1px solid #383838;
                padding: 1rem;
                margin-top: 0.5rem;
                background-color: #121212;
                color: #f8f9fa;
                text-align: center;
            ">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">Dark Mode</div>
                <div style="display: flex; justify-content: center; gap: 0.5rem;">
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #ee4347;"></div>
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #f0c84c;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Font settings
        st.markdown("<h4>Typography</h4>", unsafe_allow_html=True)
        
        font_options = ["System Default", "Sans-serif", "Serif", "Monospace"]
        current_font = SettingsState.get_setting("theme", "font_family", "System Default")
        
        selected_font = st.selectbox(
            "Font Family", 
            options=font_options,
            index=font_options.index(current_font) if current_font in font_options else 0,
            key="font_family"
        )
        
        if selected_font != current_font:
            SettingsState.set_setting("theme", "font_family", selected_font)
            
            # Apply font change via CSS
            font_css = ""
            if selected_font == "Sans-serif":
                font_css = "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important; }"
            elif selected_font == "Serif":
                font_css = "body { font-family: Georgia, 'Times New Roman', Times, serif !important; }"
            elif selected_font == "Monospace":
                font_css = "body { font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace !important; }"
            elif selected_font == "System Default":
                font_css = "body { font-family: var(--font-primary) !important; }"
            
            if font_css:
                st.markdown(f"<style>{font_css}</style>", unsafe_allow_html=True)
    
    # ===== CANVAS SETTINGS =====
    elif st.session_state.active_settings_tab == "canvas":
        st.markdown("<h3>Canvas Settings</h3>", unsafe_allow_html=True)
        
        # Canvas size
        st.markdown("<h4>Drawing Canvas</h4>", unsafe_allow_html=True)
        
        canvas_size = SettingsState.get_setting("canvas", "canvas_size", 280)
        new_canvas_size = st.slider(
            "Canvas Size",
            min_value=200,
            max_value=400,
            value=canvas_size,
            step=20,
            key="canvas_size_slider"
        )
        
        if new_canvas_size != canvas_size:
            SettingsState.set_setting("canvas", "canvas_size", new_canvas_size)
            # Reset canvas key to ensure it reloads with new size
            if "canvas_key" in st.session_state:
                import time
                st.session_state.canvas_key = f"canvas_{hash(time.time())}"
        
        # Stroke width
        stroke_width = SettingsState.get_setting("canvas", "stroke_width", 15)
        new_stroke_width = st.slider(
            "Default Stroke Width",
            min_value=5,
            max_value=30,
            value=stroke_width,
            step=1,
            key="stroke_width_slider"
        )
        
        if new_stroke_width != stroke_width:
            SettingsState.set_setting("canvas", "stroke_width", new_stroke_width)
        
        # Colors
        st.markdown("<h4>Colors</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            stroke_color = SettingsState.get_setting("canvas", "stroke_color", "#000000")
            new_stroke_color = st.color_picker(
                "Stroke Color",
                value=stroke_color,
                key="stroke_color_picker"
            )
            
            if new_stroke_color != stroke_color:
                SettingsState.set_setting("canvas", "stroke_color", new_stroke_color)
        
        with col2:
            bg_color = SettingsState.get_setting("canvas", "background_color", "#FFFFFF")
            new_bg_color = st.color_picker(
                "Background Color",
                value=bg_color,
                key="bg_color_picker"
            )
            
            if new_bg_color != bg_color:
                SettingsState.set_setting("canvas", "background_color", new_bg_color)
        
        # Grid settings
        enable_grid = SettingsState.get_setting("canvas", "enable_grid", False)
        new_enable_grid = st.toggle("Show Grid on Canvas", value=enable_grid, key="grid_toggle")
        
        if new_enable_grid != enable_grid:
            SettingsState.set_setting("canvas", "enable_grid", new_enable_grid)
            # Apply grid CSS
            if new_enable_grid:
                grid_css = """
                .canvas-container canvas {
                    background-image: linear-gradient(#ddd 1px, transparent 1px), 
                                      linear-gradient(90deg, #ddd 1px, transparent 1px);
                    background-size: 20px 20px;
                }
                """
                st.markdown(f"<style>{grid_css}</style>", unsafe_allow_html=True)
    
    # ===== PREDICTION SETTINGS =====
    elif st.session_state.active_settings_tab == "prediction":
        st.markdown("<h3>Prediction Settings</h3>", unsafe_allow_html=True)
        
        # Auto-predict
        st.markdown("<h4>Prediction Behavior</h4>", unsafe_allow_html=True)
        
        auto_predict = SettingsState.get_setting("prediction", "auto_predict", False)
        new_auto_predict = st.toggle("Auto-predict after drawing", value=auto_predict, key="auto_predict_toggle")
        
        if new_auto_predict != auto_predict:
            SettingsState.set_setting("prediction", "auto_predict", new_auto_predict)
        
        # Confidence settings
        st.markdown("<h4>Confidence Display</h4>", unsafe_allow_html=True)
        
        show_confidence = SettingsState.get_setting("prediction", "show_confidence", True)
        new_show_confidence = st.toggle("Show confidence percentage", value=show_confidence, key="confidence_toggle")
        
        if new_show_confidence != show_confidence:
            SettingsState.set_setting("prediction", "show_confidence", new_show_confidence)
        
        min_confidence = SettingsState.get_setting("prediction", "min_confidence", 0.5)
        new_min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=min_confidence,
            step=0.05,
            format="%.2f",
            key="min_confidence_slider"
        )
        
        if new_min_confidence != min_confidence:
            SettingsState.set_setting("prediction", "min_confidence", new_min_confidence)
        
        # Show alternatives
        show_alternatives = SettingsState.get_setting("prediction", "show_alternatives", True)
        new_show_alternatives = st.toggle("Show alternative predictions", value=show_alternatives, key="alternatives_toggle")
        
        if new_show_alternatives != show_alternatives:
            SettingsState.set_setting("prediction", "show_alternatives", new_show_alternatives)
    
    # ===== APP SETTINGS =====
    elif st.session_state.active_settings_tab == "app":
        st.markdown("<h3>Application Settings</h3>", unsafe_allow_html=True)
        
        # History settings
        st.markdown("<h4>History</h4>", unsafe_allow_html=True)
        
        save_history = SettingsState.get_setting("app", "save_history", True)
        new_save_history = st.toggle("Save prediction history", value=save_history, key="save_history_toggle")
        
        if new_save_history != save_history:
            SettingsState.set_setting("app", "save_history", new_save_history)
        
        max_history = SettingsState.get_setting("app", "max_history", 50)
        new_max_history = st.slider(
            "Maximum History Items",
            min_value=10,
            max_value=100,
            value=max_history,
            step=10,
            key="max_history_slider"
        )
        
        if new_max_history != max_history:
            SettingsState.set_setting("app", "max_history", new_max_history)
        
        # UI settings
        st.markdown("<h4>User Interface</h4>", unsafe_allow_html=True)
        
        show_tooltips = SettingsState.get_setting("app", "show_tooltips", True)
        new_show_tooltips = st.toggle("Show tooltips", value=show_tooltips, key="tooltips_toggle")
        
        if new_show_tooltips != show_tooltips:
            SettingsState.set_setting("app", "show_tooltips", new_show_tooltips)
            # Apply tooltip CSS
            if not new_show_tooltips:
                st.markdown("<style>[data-tooltip]{display:none !important;}</style>", unsafe_allow_html=True)
        
        debug_mode = SettingsState.get_setting("app", "debug_mode", False)
        new_debug_mode = st.toggle("Debug mode", value=debug_mode, key="debug_mode_toggle")
        
        if new_debug_mode != debug_mode:
            SettingsState.set_setting("app", "debug_mode", new_debug_mode)
    
    # Reset to defaults button
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.button("Reset All Settings to Defaults", key="reset_settings", type="secondary"):
        # Show confirmation dialog
        reset_confirmed = st.checkbox("Confirm reset - this will revert all settings to their default values", key="reset_confirm")
        
        if reset_confirmed:
            # Reset all settings categories
            SettingsState.reset_to_defaults()
            
            # Reset theme to default
            theme_manager.apply_theme(theme_manager.DEFAULT_THEME)
            
            # Force a rerun to update the UI with default values
            st.session_state.active_settings_tab = "theme"  # Reset to first tab
            st.success("All settings have been reset to defaults.")
            st.rerun()

def render_history_view():
    """
    Render the history view showing past predictions with filtering and sorting options.
    
    Features:
    - Grid display of prediction history
    - Filtering by date, prediction, confidence
    - Sorting options
    - Pagination for large history sets
    - Ability to view details or delete entries
    """
    from core.app_state.history_state import HistoryState
    import datetime
    import pandas as pd
    
    load_view_css("history")
    
    st.markdown("<h2>Prediction History</h2>", unsafe_allow_html=True)
    st.markdown("<p>View and manage your past digit predictions.</p>", unsafe_allow_html=True)
    
    # Initialize history state if needed
    if not hasattr(st.session_state, 'history_filter_date'):
        st.session_state.history_filter_date = None
    if not hasattr(st.session_state, 'history_filter_digit'):
        st.session_state.history_filter_digit = None
    if not hasattr(st.session_state, 'history_filter_min_confidence'):
        st.session_state.history_filter_min_confidence = 0.0
    if not hasattr(st.session_state, 'history_sort_by'):
        st.session_state.history_sort_by = "newest"
    if not hasattr(st.session_state, 'history_page'):
        st.session_state.history_page = 1
    if not hasattr(st.session_state, 'history_items_per_page'):
        st.session_state.history_items_per_page = 12
    
    # Get all predictions from history state
    all_predictions = HistoryState.get_predictions()
    
    # If no predictions available, show empty state
    if not all_predictions:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background-color: var(--color-background-alt); border-radius: 8px; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3>No Prediction History Yet</h3>
            <p>Your prediction history will appear here once you make some predictions.</p>
            <p>Go to the Draw view to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a button to navigate to draw view
        if st.button("Go to Draw View", key="go_to_draw", type="primary"):
            from core.app_state.navigation_state import NavigationState
            NavigationState.set_active_view("draw")
            st.rerun()
        
        return
    
    # Convert predictions to DataFrame for easier filtering and sorting
    history_data = []
    for pred in all_predictions:
        # Extract timestamp
        if isinstance(pred.get('timestamp'), str):
            try:
                timestamp = datetime.datetime.fromisoformat(pred['timestamp'])
            except (ValueError, TypeError):
                timestamp = datetime.datetime.now()
        elif isinstance(pred.get('timestamp'), datetime.datetime):
            timestamp = pred['timestamp']
        else:
            timestamp = datetime.datetime.now()
            
        # Extract other data
        history_data.append({
            'id': pred.get('id', ''),
            'digit': pred.get('digit', 0),
            'confidence': pred.get('confidence', 0.0),
            'timestamp': timestamp,
            'corrected_digit': pred.get('user_correction'),
            'image': pred.get('image'),
            'input_type': pred.get('input_type', 'canvas')
        })
    
    df = pd.DataFrame(history_data)
    
    # Create filter and sort controls
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        digit_filter = st.selectbox(
            "Filter by Digit",
            options=[None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: "All Digits" if x is None else str(x),
            key="digit_filter"
        )
        st.session_state.history_filter_digit = digit_filter
    
    with filter_col2:
        confidence_filter = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.history_filter_min_confidence,
            step=0.1,
            format="%.1f",
            key="confidence_filter"
        )
        st.session_state.history_filter_min_confidence = confidence_filter
    
    with filter_col3:
        sort_options = {
            "newest": "Newest First",
            "oldest": "Oldest First",
            "highest_conf": "Highest Confidence",
            "lowest_conf": "Lowest Confidence"
        }
        sort_by = st.selectbox(
            "Sort By",
            options=list(sort_options.keys()),
            format_func=lambda x: sort_options[x],
            key="sort_by"
        )
        st.session_state.history_sort_by = sort_by
    
    # Apply filters
    filtered_df = df.copy()
    
    if st.session_state.history_filter_digit is not None:
        filtered_df = filtered_df[filtered_df['digit'] == st.session_state.history_filter_digit]
    
    if st.session_state.history_filter_min_confidence > 0:
        filtered_df = filtered_df[filtered_df['confidence'] >= st.session_state.history_filter_min_confidence]
    
    # Apply sorting
    if st.session_state.history_sort_by == "newest":
        filtered_df = filtered_df.sort_values('timestamp', ascending=False)
    elif st.session_state.history_sort_by == "oldest":
        filtered_df = filtered_df.sort_values('timestamp', ascending=True)
    elif st.session_state.history_sort_by == "highest_conf":
        filtered_df = filtered_df.sort_values('confidence', ascending=False)
    elif st.session_state.history_sort_by == "lowest_conf":
        filtered_df = filtered_df.sort_values('confidence', ascending=True)
    
    # Reset page number if filters changed
    if (st.session_state.get('prev_digit_filter') != st.session_state.history_filter_digit or
        st.session_state.get('prev_confidence_filter') != st.session_state.history_filter_min_confidence):
        st.session_state.history_page = 1
    
    # Update previous filter values
    st.session_state.prev_digit_filter = st.session_state.history_filter_digit
    st.session_state.prev_confidence_filter = st.session_state.history_filter_min_confidence
    
    # Calculate pagination
    total_items = len(filtered_df)
    items_per_page = st.session_state.history_items_per_page
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    # Ensure current page is valid
    st.session_state.history_page = min(max(1, st.session_state.history_page), total_pages)
    
    # Select items for current page
    start_idx = (st.session_state.history_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_items = filtered_df.iloc[start_idx:end_idx]
    
    # Display stats and pagination info
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
        <div>
            <span>Showing {start_idx + 1}-{end_idx} of {total_items} predictions</span>
        </div>
        <div>
            <span>Page {st.session_state.history_page} of {total_pages}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display history entries in a grid
    if not page_items.empty:
        # Create grid layout with 3 columns
        num_items = len(page_items)
        rows = (num_items + 2) // 3  # Ceiling division
        
        for row in range(rows):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < num_items:
                    item = page_items.iloc[idx]
                    with cols[col]:
                        # Format timestamp
                        timestamp_str = item['timestamp'].strftime("%b %d, %Y %H:%M")
                        
                        # Format confidence
                        confidence_pct = f"{item['confidence'] * 100:.1f}%"
                        
                        # Create card with prediction info
                        st.markdown(f"""
                        <div style="border: 1px solid var(--color-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: var(--color-card);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <div style="font-size: 0.8rem; color: var(--color-text-light);">{timestamp_str}</div>
                                <div style="font-size: 0.8rem; color: var(--color-text-light);"><span class="highlight">Confidence: {confidence_pct}</span></div>
                            </div>
                            <div style="display: flex; gap: 1rem; align-items: center;">
                                <div style="width: 80px; height: 80px; display: flex; justify-content: center; align-items: center; background-color: var(--color-background); border-radius: 4px;">
                                    <span style="font-size: 2.5rem; font-weight: bold; color: var(--color-primary);">{item['digit']}</span>
                                </div>
                                <div>
                                    <div style="font-weight: bold; margin-bottom: 0.25rem;">Prediction: {item['digit']}</div>
                                    {f'<div style="color: var(--color-success); font-size: 0.9rem;">Corrected to: {item["corrected_digit"]}</div>' if item["corrected_digit"] is not None else ''}
                                    <div style="font-size: 0.9rem; color: var(--color-text-light);">Input: {item["input_type"].capitalize()}</div>
                                </div>
                            </div>
                            <div style="display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                                <button
                                    onclick="Streamlit.setComponentValue({{action: 'delete', id: '{item['id']}'}});"
                                    style="background: none; border: none; cursor: pointer; color: var(--color-error); font-size: 0.8rem;"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Pagination controls
    pagination_cols = st.columns([1, 3, 1])
    
    with pagination_cols[0]:
        if st.button("← Previous", key="prev_page", disabled=st.session_state.history_page <= 1):
            st.session_state.history_page -= 1
            st.rerun()
    
    with pagination_cols[1]:
        # Page selector
        page_numbers = list(range(1, total_pages + 1))
        selected_page = st.select_slider(
            "",
            options=page_numbers,
            value=st.session_state.history_page,
            key="page_selector",
            label_visibility="collapsed"
        )
        
        if selected_page != st.session_state.history_page:
            st.session_state.history_page = selected_page
            st.rerun()
    
    with pagination_cols[2]:
        if st.button("Next →", key="next_page", disabled=st.session_state.history_page >= total_pages):
            st.session_state.history_page += 1
            st.rerun()
    
    # Clear history button
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.button("Clear All History", key="clear_history", type="secondary"):
        # Show confirmation checkbox
        confirm_clear = st.checkbox("Confirm that you want to delete all prediction history", key="confirm_clear")
        
        if confirm_clear:
            # Clear history and reset filters
            HistoryState.clear_history()
            st.session_state.history_filter_date = None
            st.session_state.history_filter_digit = None
            st.session_state.history_filter_min_confidence = 0.0
            st.session_state.history_sort_by = "newest"
            st.session_state.history_page = 1
            st.success("Prediction history cleared successfully.")
            st.rerun()
    
    # Handle individual delete actions
    if "delete_id" in st.session_state and st.session_state.delete_id:
        entry_id = st.session_state.delete_id
        # Delete the entry
        # In a real implementation, you would call a method to delete specific entries
        # For example: HistoryState.delete_entry(entry_id)
        st.session_state.delete_id = None
        st.rerun()
    
    # Add JavaScript to handle delete buttons
    st.markdown("""
    <script>
    // Listen for messages from component
    window.addEventListener('message', function(event) {
        if (event.data.type === 'streamlit:componentOutput') {
            const data = event.data.value;
            if (data && data.action === 'delete') {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: {
                        delete_id: data.id
                    }
                }, '*');
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)


# def render_sidebar():
#     """
#     Render a unified sidebar with navigation and theme toggle.
#     """
#     from ui.theme.theme_manager import theme_manager
#     from core.app_state.navigation_state import NavigationState
#     import datetime
#     from ui.components.controls.bb8_toggle import BB8Toggle
    
#     # Render header
#     st.sidebar.markdown(
#         """
#         <div class="sidebar-header">
#             <div class="gradient-text">MNIST App</div>
#             <div class="sidebar-subheader">Digit Classification AI</div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
    
#     # Navigation items
#     nav_items = [
#         {"id": "home", "label": "Home", "icon": "🏠"},
#         {"id": "draw", "label": "Draw", "icon": "✏️"},
#         {"id": "history", "label": "History", "icon": "📊"},
#         {"id": "settings", "label": "Settings", "icon": "⚙️"}
#     ]
    
#     # Get current view
#     active_view = NavigationState.get_active_view()
    
#     # Add some spacing
#     st.sidebar.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    
#     # Create navigation buttons
#     for item in nav_items:
#         if st.sidebar.button(
#             f"{item['icon']} {item['label']}",
#             key=f"nav_{item['id']}",
#             type="primary" if active_view == item['id'] else "secondary",
#             use_container_width=True
#         ):
#             NavigationState.set_active_view(item['id'])
#             st.rerun()
    
#     # Add divider
#     st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
#     # BB8 Toggle Container
#     with st.sidebar.container():
#         # Create and display the BB8 toggle
#         bb8_key = "sidebar_bb8_toggle"
#         bb8_toggle = BB8Toggle(
#             theme_manager_instance=theme_manager, 
#             on_change=lambda theme: st.rerun(),
#             key=bb8_key
#         )
#         toggle_result = bb8_toggle.display()
    
#     # Add divider
#     st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
#     # Add version in footer
#     current_year = datetime.datetime.now().year
#     version = "1.0.0"  # Can be fetched from a config
    
#     st.sidebar.markdown(
#         f"""
#         <div class="sidebar-footer">
#             <p>Version {version}</p>
#             <p>© {current_year} MNIST Classifier</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )
def render_sidebar():
    """
    Render a unified sidebar with navigation and theme toggle.
    """
    from ui.theme.theme_manager import theme_manager
    from core.app_state.navigation_state import NavigationState
    import datetime
    # Import your revised BB8Toggle component:
    from ui.components.controls.bb8_toggle import BB8Toggle
    
    # Sidebar header
    st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <div class="gradient-text">MNIST App</div>
            <div class="sidebar-subheader">Digit Classification AI</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Navigation items
    nav_items = [
        {"id": "home", "label": "Home", "icon": "🏠"},
        {"id": "draw", "label": "Draw", "icon": "✏️"},
        {"id": "history", "label": "History", "icon": "📊"},
        {"id": "settings", "label": "Settings", "icon": "⚙️"}
    ]
    
    active_view = NavigationState.get_active_view()
    
    # Spacer
    st.sidebar.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
    
    # Create navigation buttons
    for item in nav_items:
        if st.sidebar.button(
            f"{item['icon']} {item['label']}",
            key=f"nav_{item['id']}",
            type="primary" if active_view == item['id'] else "secondary",
            use_container_width=True
        ):
            NavigationState.set_active_view(item['id'])
            st.rerun()
    
    # Divider
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # BB8 Toggle Container
    with st.sidebar.container():
        # Instantiate BB8Toggle
        bb8_toggle = BB8Toggle(
            theme_manager_instance=theme_manager,
            on_change=lambda new_theme: st.rerun(),  # reload to apply new theme
            key="sidebar_bb8_toggle"
        )
        # Display the toggle; it handles theme switching internally
        bb8_toggle.display()
    
    # Another divider
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Footer with version/year
    current_year = datetime.datetime.now().year
    version = "1.0.0"  # Or load from config
    
    st.sidebar.markdown(
        f"""
        <div class="sidebar-footer">
            <p>Version {version}</p>
            <p>© {current_year} MNIST Classifier</p>
        </div>
        """,
        unsafe_allow_html=True
    )

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
    
    # Render the custom sidebar instead of using Layout's sidebar
    render_sidebar()

    # Set up layout with sidebar=False to prevent double rendering
    layout = Layout(title="MNIST Digit Classifier")
    
    # Render header only (not sidebar)
    layout.render_header()
    
    
    # Determine which view to render based on navigation state
    current_view = NavigationState.get_active_view()
    
    # Render the appropriate view content
    if current_view == "home":
        render_home_view()
    elif current_view == "draw":
        render_draw_view()
    elif current_view == "history":
        render_history_view()
    elif current_view == "settings":
        render_settings_view()

    # Render footer last
    layout.render_footer()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("The application encountered a critical error. Please check the logs for details.")
        st.code(str(e))


