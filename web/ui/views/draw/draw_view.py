# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/draw/draw_view.py
# Description: Drawing view for digit classification
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
import base64
from PIL import Image
import io
import random

from ui.views.base_view import BaseView
from core.app_state.canvas_state import CanvasState
from core.app_state.history_state import HistoryState

logger = logging.getLogger(__name__)

class DrawView(BaseView):
    """Drawing view for digit classification.
    
    This view allows users to draw digits, upload images, or provide URLs
    for digit classification.
    """
    
    def __init__(self):
        """Initialize the drawing view."""
        super().__init__(
            view_id="draw",
            title="Draw a Digit",
            description="Draw a digit to classify it using machine learning",
            icon="✏️"
        )
        self.logger = logging.getLogger(__name__)
    
    def render(self) -> None:
        """Render the drawing view content."""
        self.logger.debug("Rendering drawing view")
        
        try:
            # Apply common layout styling
            self.apply_common_layout()
            
            # Create welcome content card
            welcome_content = """
            <p>Draw a digit (0-9) in the box below, and our model will predict which digit you've drawn.</p>
            <p>You can also upload an image or provide a URL to an image of a handwritten digit.</p>
            """
            
            st.markdown(self._create_welcome_card(
                "Draw a Digit", "✏️", welcome_content
            ), unsafe_allow_html=True)
            
            # Main drawing area
            st.write("### Draw Here")
            
            # Create a simple canvas using Streamlit's native drawing capabilities
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Canvas placeholder - in a real implementation, use streamlit_drawable_canvas
                st.markdown("""
                <div style="
                    width: 280px;
                    height: 280px;
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                ">
                    <p style="color: #888888; font-size: 1.2em;">
                        Canvas will appear here
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("### Controls")
                
                # Control buttons
                if st.button("🧹 Clear Canvas", use_container_width=True):
                    st.session_state["canvas_data"] = None
                    st.success("Canvas cleared!")
                
                if st.button("🔍 Predict", use_container_width=True, type="primary"):
                    self._perform_mock_prediction()
            
            # Show the prediction result if available
            if "prediction_result" in st.session_state:
                self._display_prediction_result(st.session_state["prediction_result"])
            
            # Close the view container
            st.markdown('</div>', unsafe_allow_html=True)
            
            self.logger.debug("Draw view rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering draw view: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
    
    def apply_common_layout(self):
        """Apply common layout styling for the view."""
        st.markdown("""
        <style>
        /* Fix content alignment */
        .block-container {
            max-width: 100% !important;
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* View container */
        .view-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Canvas styling */
        .canvas-container {
            margin-bottom: 1rem;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Prediction result */
        .prediction-result {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        
        .prediction-digit {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #4F46E5;
        }
        
        .confidence-bar {
            height: 20px;
            background-color: #4F46E5;
            border-radius: 4px;
            margin-top: 0.5rem;
        }
        
        /* Dark mode adjustments */
        [data-theme="dark"] .prediction-result {
            background-color: #2d3748;
            border-color: #4a5568;
        }
        
        [data-theme="dark"] .prediction-digit {
            color: #818cf8;
        }
        </style>
        
        <div class="view-container">
        """, unsafe_allow_html=True)
    
    def _create_welcome_card(self, title, icon, content):
        """Create a welcome card with standard styling."""
        # Format content to ensure paragraphs are properly wrapped
        if not content.startswith('<p>'):
            paragraphs = content.split('\n')
            content = ''.join([f'<p>{p.strip()}</p>' for p in paragraphs if p.strip()])
        
        return f"""
        <div class="card card-elevated content-card welcome-card animate-fade-in">
            <div class="card-title">
                <span class="card-icon">{icon}</span>
                {title}
            </div>
            <div class="card-content">
                {content}
            </div>
        </div>
        """
    
    def _perform_mock_prediction(self):
        """Perform a mock prediction for demo purposes."""
        try:
            # Generate a random prediction
            digit = random.randint(0, 9)
            confidence = random.uniform(0.75, 0.99)
            
            # Create a result dictionary
            result = {
                "digit": digit,
                "confidence": confidence,
                "probabilities": {
                    str(i): confidence if i == digit else random.uniform(0, 0.1) 
                    for i in range(10)
                }
            }
            
            # Normalize probabilities
            total = sum(result["probabilities"].values())
            result["probabilities"] = {
                k: v / total for k, v in result["probabilities"].items()
            }
            
            # Store in session state
            st.session_state["prediction_result"] = result
            
            # Add to history
            try:
                # We'd normally use actual image data here
                mock_image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                HistoryState.add_prediction(digit, confidence, mock_image_data)
            except Exception as e:
                self.logger.error(f"Error adding to history: {str(e)}")
                
            st.experimental_rerun()
        except Exception as e:
            self.logger.error(f"Error in mock prediction: {str(e)}")
            st.error(f"Error generating prediction: {str(e)}")
    
    def _display_prediction_result(self, result):
        """Display the prediction result."""
        st.markdown("### Prediction Result")
        
        # Create a nice display for the prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-result">
                <div class="prediction-digit">{result['digit']}</div>
                <p style="text-align: center;">Confidence: {result['confidence']:.2%}</p>
                <div class="confidence-bar" style="width: {result['confidence'] * 100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Probability Distribution")
            
            # Display all probabilities
            for digit, prob in result["probabilities"].items():
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="width: 20px; text-align: right; margin-right: 10px;">{digit}</div>
                    <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 4px; height: 12px;">
                        <div style="width: {prob * 100}%; background-color: #4F46E5; height: 12px; border-radius: 4px;"></div>
                    </div>
                    <div style="width: 60px; text-align: right; margin-left: 10px;">{prob:.2%}</div>
                </div>
                """, unsafe_allow_html=True) 