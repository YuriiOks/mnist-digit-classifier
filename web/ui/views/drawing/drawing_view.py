# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/drawing/drawing_view.py
# Description: Drawing view for digit classification
# Created: 2024-05-01

# This file provides compatibility for the "drawing" view path referenced in app.py
# It simply imports and re-exports the DrawView from the "draw" package

from ui.views.draw.draw_view import DrawView as DrawingView

# Keep the original class name for compatibility
__all__ = ['DrawingView']

import streamlit as st
import logging
import numpy as np
from typing import Dict, Any, List, Optional
import io
import base64
from PIL import Image

from ui.views.base_view import BaseView
from ui.components.cards.content_card import ContentCard
from ui.components.inputs.canvas import DrawingCanvas
from ui.components.inputs.file_upload import ImageUpload
from ui.components.inputs.url_input import ImageUrlInput
from ui.components.navigation.tabs import InputTabs
from ui.components.controls.buttons import PrimaryButton, SecondaryButton
from ui.components.feedback.prediction_result import PredictionResult

from core.app_state.canvas_state import CanvasState
from core.app_state.history_state import HistoryState

# This would typically be imported from a service
# For now, we'll create a mock prediction function
def predict_digit(image_data: bytes) -> Dict[str, Any]:
    """Mock function to predict digit from image data.
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        Prediction result dictionary
    """
    import random
    
    # Randomly generate a prediction (mock)
    digit = random.randint(0, 9)
    confidence = random.uniform(0.7, 0.99)
    
    # Generate random probabilities
    probabilities = {}
    for i in range(10):
        if i == digit:
            probabilities[str(i)] = confidence
        else:
            probabilities[str(i)] = random.uniform(0, (1 - confidence) / 9)
    
    # Normalize probabilities to sum to 1
    total = sum(probabilities.values())
    probabilities = {k: v / total for k, v in probabilities.items()}
    
    return {
        "digit": digit,
        "confidence": confidence,
        "probabilities": probabilities
    }

logger = logging.getLogger(__name__)

class DrawingView(BaseView):
    """Drawing view for digit classification."""
    
    def __init__(self):
        """Initialize drawing view."""
        super().__init__(
            view_id="drawing",
            title="Draw & Classify",
            description="Draw a digit or upload an image for classification",
            icon="✏️"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self) -> None:
        """Render the drawing view."""
        self.logger.debug("Rendering drawing view")
        
        try:
            # Apply common layout styling
            self.apply_common_layout()
            
            # Create welcome card
            welcome_card = ContentCard(
                title="Draw a Digit",
                icon="✏️",
                content="""
                <p>Draw a digit from 0-9 in the canvas below, or upload an image file.</p>
                <p>Our AI model will analyze your drawing and predict which digit it represents.</p>
                <p>You can also provide feedback to help improve the model's accuracy.</p>
                """,
                elevated=True,
                classes=["welcome-card"]
            )
            welcome_card.render()
            
            # Main content layout - two columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Input container
                st.markdown("<h3>Draw or Upload</h3>", unsafe_allow_html=True)
                
                # Input tabs
                input_type, tabs = InputTabs().render()
                
                # Draw tab
                with tabs[0]:
                    # Drawing canvas
                    canvas = DrawingCanvas(key="drawing_canvas")
                    canvas.render()
                    
                    # Action buttons
                    st.markdown("<div style='display: flex; gap: 10px;'>", unsafe_allow_html=True)
                    
                    # Predict button
                    predict_button = PrimaryButton(
                        label="Predict",
                        on_click=self._handle_predict_canvas,
                        key="predict_button",
                        icon="🔍"
                    )
                    predict_button.render()
                    
                    # Clear button
                    clear_button = SecondaryButton(
                        label="Clear",
                        on_click=self._handle_clear_canvas,
                        key="clear_button",
                        icon="🗑️"
                    )
                    clear_button.render()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Upload tab
                with tabs[1]:
                    # File upload component
                    file_upload = ImageUpload(key="image_upload")
                    file_upload.render()
                    
                    # Predict button for upload
                    predict_upload_button = PrimaryButton(
                        label="Predict",
                        on_click=self._handle_predict_upload,
                        key="predict_upload_button",
                        icon="🔍"
                    )
                    predict_upload_button.render()
                
                # URL tab
                with tabs[2]:
                    # URL input component
                    url_input = ImageUrlInput(key="image_url")
                    url_input.render()
                    
                    # Predict button for URL
                    predict_url_button = PrimaryButton(
                        label="Predict",
                        on_click=self._handle_predict_url,
                        key="predict_url_button",
                        icon="🔍"
                    )
                    predict_url_button.render()
            
            with col2:
                # Results container
                st.markdown("<h3>Prediction</h3>", unsafe_allow_html=True)
                
                # Prediction result
                prediction_result = PredictionResult(
                    on_correction=self._handle_correction
                )
                prediction_result.render()
                
                # Tips card
                tips_card = ContentCard(
                    title="Tips for Better Results",
                    icon="💡",
                    content="""
                    <p><strong>For best results:</strong></p>
                    <ul>
                        <li>Draw digits clearly in the center of the canvas</li>
                        <li>Use black ink on white background</li>
                        <li>Draw a single digit per prediction</li>
                        <li>Try different stroke widths if needed</li>
                    </ul>
                    """,
                    elevated=False,
                    classes=["tips-card"]
                )
                tips_card.render()
            
            self.logger.debug("Drawing view rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering drawing view: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering the drawing view: {str(e)}")
    
    def _handle_predict_canvas(self) -> None:
        """Handle prediction from canvas input."""
        self.logger.debug("Handling canvas prediction")
        
        try:
            # Check if we have canvas data
            canvas_data = CanvasState.get_canvas_data()
            if not canvas_data or canvas_data.get("is_empty", True):
                st.warning("Please draw something before predicting.")
                return
            
            # Get image data from canvas
            # In a real implementation, this would process the canvas data
            # For now, we'll mock it with a hardcoded image
            
            # Create a simple 28x28 image as mock data
            img = Image.new('L', (28, 28), color=255)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            
            # Store image data for history
            CanvasState.set_image_data(image_data)
            CanvasState.set_input_type(CanvasState.CANVAS_INPUT)
            
            # Perform prediction
            self._perform_prediction(image_data)
            
        except Exception as e:
            self.logger.error(f"Error in canvas prediction: {str(e)}", exc_info=True)
            st.error(f"An error occurred during prediction: {str(e)}")
    
    def _handle_predict_upload(self) -> None:
        """Handle prediction from uploaded image."""
        self.logger.debug("Handling upload prediction")
        
        try:
            # Check if we have image data
            image_data = CanvasState.get_image_data()
            if not image_data:
                st.warning("Please upload an image before predicting.")
                return
            
            # Perform prediction
            self._perform_prediction(image_data)
            
        except Exception as e:
            self.logger.error(f"Error in upload prediction: {str(e)}", exc_info=True)
            st.error(f"An error occurred during prediction: {str(e)}")
    
    def _handle_predict_url(self) -> None:
        """Handle prediction from URL image."""
        self.logger.debug("Handling URL prediction")
        
        try:
            # Check if we have image data
            image_data = CanvasState.get_image_data()
            if not image_data:
                st.warning("Please load an image from URL before predicting.")
                return
            
            # Perform prediction
            self._perform_prediction(image_data)
            
        except Exception as e:
            self.logger.error(f"Error in URL prediction: {str(e)}", exc_info=True)
            st.error(f"An error occurred during prediction: {str(e)}")
    
    def _handle_clear_canvas(self) -> None:
        """Handle clearing the canvas."""
        self.logger.debug("Clearing canvas")
        
        try:
            # Clear canvas state
            CanvasState.clear_all()
            # Reset current prediction
            HistoryState.initialize()
            
            # Rerun to update UI
            st.experimental_rerun()
            
        except Exception as e:
            self.logger.error(f"Error clearing canvas: {str(e)}", exc_info=True)
            st.error(f"An error occurred while clearing: {str(e)}")
    
    def _handle_correction(self, correct_digit: int) -> None:
        """Handle user correction of prediction.
        
        Args:
            correct_digit: The correct digit value (0-9)
        """
        self.logger.debug(f"Handling correction to digit: {correct_digit}")
        
        try:
            # Get current prediction
            current = HistoryState.get_current_prediction()
            if current:
                # Update with correction
                HistoryState.set_user_correction(current["id"], correct_digit)
                st.success(f"Thank you for your feedback! Recorded correct value: {correct_digit}")
                
                # Rerun to update UI
                st.experimental_rerun()
            else:
                st.warning("No current prediction to correct.")
                
        except Exception as e:
            self.logger.error(f"Error handling correction: {str(e)}", exc_info=True)
            st.error(f"An error occurred while processing correction: {str(e)}")
    
    def _perform_prediction(self, image_data: bytes) -> None:
        """Perform prediction on image data and save to history.
        
        Args:
            image_data: Image data as bytes
        """
        self.logger.debug("Performing prediction")
        
        try:
            # Call prediction function
            # In a real implementation, this would call a ML model service
            prediction_result = predict_digit(image_data)
            
            # Convert image to base64 for storage
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Add to history
            HistoryState.add_prediction(image_b64, prediction_result)
            
            # Success message
            st.success(f"Prediction complete! Predicted digit: {prediction_result['digit']}")
            
            # Rerun to update UI
            st.experimental_rerun()
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            st.error(f"An error occurred during prediction: {str(e)}") 