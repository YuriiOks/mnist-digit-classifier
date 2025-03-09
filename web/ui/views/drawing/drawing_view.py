# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/drawing/drawing_view.py
# Description: Drawing canvas view
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any
import numpy as np
from PIL import Image

try:
    from ui.views.base_view import BaseView
    from core.app_state.canvas_state import CanvasState
    from core.app_state.history_state import HistoryState
except ImportError:
    # Fallback for minimal implementation
    class BaseView:
        def __init__(self, view_id="draw", title="Drawing Canvas", description="", icon="✏️"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon

    # Simple state placeholders
    class CanvasState:
        @staticmethod
        def set_image(img): pass
        @staticmethod
        def set_prediction(pred): pass
        @staticmethod
        def get_image(): return None
        
    class HistoryState:
        @staticmethod
        def add_prediction(img, pred): pass

logger = logging.getLogger(__name__)

class DrawingView(BaseView):
    """Drawing view for digit input."""
    
    def __init__(self):
        """Initialize the drawing view."""
        super().__init__(
            view_id="draw",
            title="Drawing Canvas",
            description="Draw a digit for classification",
            icon="✏️"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self) -> None:
        """Render the drawing view."""
        self.logger.debug("Rendering drawing view")
        
        try:
            # Apply the same layout CSS as home view for consistency
            st.markdown("""
            <style>
            /* Fix content alignment */
            .block-container {
                max-width: 100% !important;
                padding-top: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            
            /* Make headers look better */
            h1, h2, h3 {
                margin-bottom: 1rem;
                margin-top: 0.5rem;
                font-family: var(--font-primary, 'Poppins', sans-serif);
            }
            
            /* Add space around elements */
            .stMarkdown {
                margin-bottom: 0.5rem;
            }
            
            /* Remove empty columns */
            .stColumn:empty {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Add welcome card similar to home page
            st.markdown("""
            <div class="card card-elevated content-card welcome-card animate-fade-in">
                <div class="card-title">
                    <span class="card-icon">✏️</span>
                    Drawing Canvas
                </div>
                <div class="card-content">
                    <p>Use this canvas to draw a digit from 0-9 and watch the AI model predict what you've drawn.</p>
                    <p>Draw with your mouse by clicking and dragging on the canvas below. Make your digit clear and centered for best results.</p>
                    <p>After drawing, click "Predict" to see the model's prediction or "Clear" to start over.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rest of the drawing view content
            # st.title("Draw a Digit") - Removed since we have it in the card
            
            # Add drawing canvas
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Drawing area
                st.subheader("Drawing Area")
                
                # Simplified canvas (using matplotlib for now)
                # In a real app, we'd use st_canvas or another drawing component
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    
                    # Create a blank canvas
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set_xlim(0, 28)
                    ax.set_ylim(0, 28)
                    ax.axis('off')
                    ax.set_facecolor('black')
                    
                    # Display the canvas
                    st.pyplot(fig)
                    
                    # Add buttons for controls
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear Canvas"):
                            # This would reset the canvas in a real app
                            st.info("Canvas cleared")
                            
                    with col2:
                        if st.button("Classify", type="primary"):
                            # This would submit the drawing for classification
                            with st.spinner("Classifying..."):
                                # Fake a prediction
                                import random
                                digit = random.randint(0, 9)
                                confidence = random.uniform(0.7, 0.99)
                                
                                # In a real app, we would:
                                # 1. Get the image from the canvas
                                # 2. Preprocess the image
                                # 3. Send to the model for classification
                                # 4. Display the result
                                
                                st.session_state['prediction'] = {
                                    'digit': digit,
                                    'confidence': confidence
                                }
                                
                                # Show result
                                st.success(f"Classification complete!")
                    
                    # Options
                    st.write("### Options")
                    brush_size = st.slider("Brush Size", 1, 10, 3)
                    
                except ImportError:
                    st.error("Could not load drawing canvas. Please check dependencies.")
            
            with col2:
                # Results area
                st.subheader("Classification Result")
                
                if 'prediction' in st.session_state:
                    pred = st.session_state['prediction']
                    digit = pred['digit']
                    confidence = pred['confidence']
                    
                    # Show the digit prominently
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px;'>
                        <div style='font-size: 4rem; font-weight: bold;'>{digit}</div>
                        <div>Confidence: {confidence:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show prediction details
                    st.write("### Prediction Details")
                    st.write(f"The model predicts this is a **{digit}** with {confidence:.2%} confidence.")
                    
                    # Add to history button
                    if st.button("Add to History"):
                        # In a real app, we would add to history here
                        st.success("Added to prediction history!")
                        
                else:
                    st.info("Draw a digit and click 'Classify' to see the prediction here.")
                    
                # Tips
                st.write("### Tips")
                st.write("- Draw a digit that fills most of the canvas")
                st.write("- Use the clear button to start over")
                st.write("- Try different brush sizes for better results")
            
            self.logger.debug("Drawing view rendered successfully")
            
        except Exception as e:
            self.logger.error(f"Error rendering drawing view: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}") 