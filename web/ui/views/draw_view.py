# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/draw_view.py
# Description: Draw view implementation
# Created: 2025-03-17

import streamlit as st
import time
import random
import logging
from typing import Optional
from PIL import Image
import io

from ui.views.base_view import View
from core.app_state.canvas_state import CanvasState
from core.app_state.settings_state import SettingsState

class DrawView(View):
    """Draw view for the MNIST Digit Classifier application."""
    
    def __init__(self):
        """Initialize the draw view."""
        super().__init__(
            name="draw",
            title="Digit Recognition",
            description="Choose a method to input a digit for recognition."
        )
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for the draw view."""
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
            
    def _render_tab_buttons(self) -> None:
        """Render tab selection buttons."""
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
    
    def _render_draw_tab(self) -> None:
        """Render the draw digit tab content."""
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
    
    def _render_upload_tab(self) -> None:
        """Render the upload image tab content."""
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
    
    def _render_url_tab(self) -> None:
        """Render the URL input tab content."""
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
                
    def _render_action_buttons(self) -> None:
        """Render action buttons (Clear, Predict)."""
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
                
    def _render_prediction_result(self) -> None:
        """Render prediction result if available."""
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
    
    def render(self) -> None:
        """Render the draw view content."""
        # Initialize session state variables
        self._initialize_session_state()
        
        # Tab navigation
        self._render_tab_buttons()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Input container based on active tab
        if st.session_state.active_tab == "draw":
            self._render_draw_tab()
        elif st.session_state.active_tab == "upload":
            self._render_upload_tab()
        elif st.session_state.active_tab == "url":
            self._render_url_tab()
        
        # Action buttons - always visible regardless of active tab
        st.markdown("<hr>", unsafe_allow_html=True)
        self._render_action_buttons()
        
        # Prediction result
        self._render_prediction_result()