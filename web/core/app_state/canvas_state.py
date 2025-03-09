# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/canvas_state.py
# Description: State management for canvas and drawing
# Created: 2024-05-01

import logging
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import base64
from PIL import Image
import io

from core.app_state.session_state import SessionState

logger = logging.getLogger(__name__)

class CanvasState:
    """Manage canvas and drawing state."""
    
    CANVAS_KEY = "canvas_data"
    INPUT_TYPE_KEY = "input_type"  # canvas, upload, url
    IMAGE_DATA_KEY = "image_data"  # Stores actual image data
    URL_KEY = "image_url"
    
    # Input types
    CANVAS_INPUT = "canvas"
    UPLOAD_INPUT = "upload"
    URL_INPUT = "url"
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize canvas state with default values."""
        if SessionState.get(cls.CANVAS_KEY) is None:
            logger.debug("Initializing default canvas state")
            SessionState.set(cls.CANVAS_KEY, {
                "strokes": [],
                "is_empty": True,
                "last_updated": None
            })
        
        if SessionState.get(cls.INPUT_TYPE_KEY) is None:
            logger.debug("Setting default input type")
            SessionState.set(cls.INPUT_TYPE_KEY, cls.CANVAS_INPUT)
    
    @classmethod
    def get_input_type(cls) -> str:
        """Get current input type.
        
        Returns:
            Current input type (canvas, upload, url)
        """
        cls.initialize()
        return SessionState.get(cls.INPUT_TYPE_KEY)
    
    @classmethod
    def set_input_type(cls, input_type: str) -> None:
        """Set input type.
        
        Args:
            input_type: Input type to set (canvas, upload, url)
        """
        if input_type not in [cls.CANVAS_INPUT, cls.UPLOAD_INPUT, cls.URL_INPUT]:
            logger.warning(f"Invalid input type: {input_type}")
            return
            
        cls.initialize()
        SessionState.set(cls.INPUT_TYPE_KEY, input_type)
        logger.debug(f"Input type set to: {input_type}")
    
    @classmethod
    def get_canvas_data(cls) -> Dict[str, Any]:
        """Get current canvas data.
        
        Returns:
            Dictionary with canvas stroke data
        """
        cls.initialize()
        return SessionState.get(cls.CANVAS_KEY)
    
    @classmethod
    def update_canvas_data(cls, strokes: List[Dict[str, Any]]) -> None:
        """Update canvas strokes data.
        
        Args:
            strokes: List of stroke data dictionaries
        """
        from datetime import datetime
        
        cls.initialize()
        canvas_data = {
            "strokes": strokes,
            "is_empty": not bool(strokes),
            "last_updated": datetime.now().isoformat()
        }
        SessionState.set(cls.CANVAS_KEY, canvas_data)
        logger.debug("Canvas data updated")
    
    @classmethod
    def clear_canvas(cls) -> None:
        """Clear canvas data."""
        cls.initialize()
        canvas_data = {
            "strokes": [],
            "is_empty": True,
            "last_updated": None
        }
        SessionState.set(cls.CANVAS_KEY, canvas_data)
        logger.debug("Canvas cleared")
    
    @classmethod
    def set_image_data(cls, image_data: Union[bytes, np.ndarray, Image.Image]) -> None:
        """Set image data.
        
        Args:
            image_data: Image data as bytes, numpy array, or PIL Image
        """
        cls.initialize()
        
        # Convert to bytes if needed
        if isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image
            img = Image.fromarray(image_data.astype('uint8'))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
        elif isinstance(image_data, Image.Image):
            buffer = io.BytesIO()
            image_data.save(buffer, format="PNG")
            image_data = buffer.getvalue()
        
        # Store as base64 string
        if isinstance(image_data, bytes):
            encoded = base64.b64encode(image_data).decode('utf-8')
            SessionState.set(cls.IMAGE_DATA_KEY, encoded)
            logger.debug("Image data set")
        else:
            logger.warning(f"Unsupported image data type: {type(image_data)}")
    
    @classmethod
    def get_image_data(cls) -> Optional[bytes]:
        """Get image data.
        
        Returns:
            Image data as bytes or None if not set
        """
        cls.initialize()
        encoded = SessionState.get(cls.IMAGE_DATA_KEY)
        if encoded:
            return base64.b64decode(encoded)
        return None
    
    @classmethod
    def set_image_url(cls, url: str) -> None:
        """Set image URL.
        
        Args:
            url: URL to image
        """
        cls.initialize()
        SessionState.set(cls.URL_KEY, url)
        logger.debug(f"Image URL set: {url}")
    
    @classmethod
    def get_image_url(cls) -> Optional[str]:
        """Get image URL.
        
        Returns:
            Image URL or None if not set
        """
        cls.initialize()
        return SessionState.get(cls.URL_KEY)
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all image data (canvas, upload, URL)."""
        cls.clear_canvas()
        SessionState.set(cls.IMAGE_DATA_KEY, None)
        SessionState.set(cls.URL_KEY, None)
        logger.debug("All image data cleared") 