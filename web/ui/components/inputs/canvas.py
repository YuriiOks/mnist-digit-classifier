# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/inputs/canvas.py
# Description: Drawing canvas component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple
import base64
from PIL import Image
import io
import numpy as np

from ui.components.base.component import Component
from core.app_state.canvas_state import CanvasState
from core.app_state.settings_state import SettingsState

logger = logging.getLogger(__name__)

class DrawingCanvas(Component):
    """Drawing canvas component for digit input."""
    
    def __init__(
        self,
        key: str = "drawing_canvas",
        on_change: Optional[Callable] = None,
        classes: List[str] = None,
        attributes: Dict[str, str] = None
    ):
        """Initialize drawing canvas component.
        
        Args:
            key: Unique key for the component
            on_change: Callback for canvas changes
            classes: Additional CSS classes
            attributes: Additional HTML attributes
        """
        super().__init__(classes=classes, attributes=attributes)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.key = key
        self.on_change = on_change
        
    def render(self) -> None:
        """Render the drawing canvas component."""
        self.logger.debug("Rendering drawing canvas")
        
        try:
            # Get canvas settings
            canvas_size = SettingsState.get_setting("canvas", "canvas_size", 280)
            stroke_width = SettingsState.get_setting("canvas", "stroke_width", 15)
            stroke_color = SettingsState.get_setting("canvas", "stroke_color", "#000000")
            background_color = SettingsState.get_setting("canvas", "background_color", "#FFFFFF")
            
            # Create column layout with canvas in the middle
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col2:
                # Create the canvas with streamlit-drawable-canvas
                try:
                    from streamlit_drawable_canvas import st_canvas
                    
                    # Custom CSS for canvas
                    st.markdown("""
                    <style>
                    .canvas-instructions {
                        text-align: center;
                        margin-bottom: 1rem;
                        font-size: 0.9rem;
                        color: #666;
                    }
                    
                    .canvas-container canvas {
                        border: 2px solid #e0e0e0;
                        border-radius: 4px;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Instructions
                    st.markdown("""
                    <div class="canvas-instructions">
                        <p>Draw a digit from 0-9 in the center of the canvas</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Render canvas with proper settings
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.0)",
                        stroke_width=stroke_width,
                        stroke_color=stroke_color,
                        background_color=background_color,
                        height=canvas_size,
                        width=canvas_size,
                        drawing_mode="freedraw",
                        key=self.key,
                        display_toolbar=True,
                        update_streamlit=True,
                    )
                    
                    # Handle canvas changes
                    if canvas_result.json_data is not None:
                        # Store data in state
                        if "objects" in canvas_result.json_data:
                            strokes = canvas_result.json_data["objects"]
                            CanvasState.update_canvas_data(strokes)
                            
                            # Convert image data if there are strokes
                            if len(strokes) > 0 and canvas_result.image_data is not None:
                                # Convert to grayscale and save to state
                                img_array = canvas_result.image_data
                                img = Image.fromarray(img_array).convert('L')
                                buffer = io.BytesIO()
                                img.save(buffer, format="PNG")
                                img_bytes = buffer.getvalue()
                                CanvasState.set_image_data(img_bytes)
                                
                                # Set input type to canvas
                                CanvasState.set_input_type(CanvasState.CANVAS_INPUT)
                                
                                # Call on_change callback if provided
                                if self.on_change:
                                    self.on_change()
                    
                except ImportError:
                    st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
            
        except Exception as e:
            self.logger.error(f"Error rendering drawing canvas: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering the drawing canvas: {str(e)}") 