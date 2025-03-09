# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/canvas_state.py
# Description: Canvas-specific state management
# Created: 2024-05-01

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import json
from pathlib import Path
import base64
import io
import logging
import numpy as np
from io import BytesIO
from PIL import Image

from core.app_state.session_state import SessionState

logger = logging.getLogger(__name__)

class CanvasState:
    """Canvas-specific state management.
    
    This class manages the drawing canvas state, including drawings,
    settings, and related functionality.
    """
    
    # Canvas state key prefixes
    PREFIX = "canvas_"
    CURRENT_DRAWING_KEY = f"{PREFIX}current_drawing"
    DRAWING_DATA_KEY = f"{PREFIX}drawing_data"
    SETTINGS_KEY = f"{PREFIX}settings"
    PREDICTION_KEY = f"{PREFIX}prediction"
    IS_DRAWING_KEY = f"{PREFIX}is_drawing"
    LAST_CHANGE_KEY = f"{PREFIX}last_change"
    
    # State keys
    CANVAS_DATA_KEY = "_canvas_data"
    BRUSH_SIZE_KEY = "_brush_size"
    BRUSH_COLOR_KEY = "_brush_color"
    CANVAS_WIDTH_KEY = "_canvas_width"
    CANVAS_HEIGHT_KEY = "_canvas_height"
    CURRENT_IMAGE_KEY = "_current_image"
    
    # Default canvas settings
    DEFAULT_SETTINGS = {
        "stroke_width": 15,
        "stroke_color": "#000000",
        "background_color": "#ffffff",
        "drawing_mode": "freedraw",
        "canvas_width": 280,
        "canvas_height": 280,
        "realtime_update": True
    }
    
    # Default values
    DEFAULT_BRUSH_SIZE = 15
    DEFAULT_BRUSH_COLOR = "black"
    DEFAULT_CANVAS_WIDTH = 280
    DEFAULT_CANVAS_HEIGHT = 280
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize canvas state with default values if not already set."""
        logger.debug("Initializing canvas state")
        try:
            # Initialize settings
            if not SessionState.has_key(cls.SETTINGS_KEY):
                logger.debug("Setting default settings")
                SessionState.set(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            
            # Initialize current drawing
            if not SessionState.has_key(cls.CURRENT_DRAWING_KEY):
                logger.debug("Setting default current drawing")
                SessionState.set(cls.CURRENT_DRAWING_KEY, None)
            
            # Initialize drawing data
            if not SessionState.has_key(cls.DRAWING_DATA_KEY):
                logger.debug("Setting default drawing data")
                SessionState.set(cls.DRAWING_DATA_KEY, None)
            
            # Initialize prediction data
            if not SessionState.has_key(cls.PREDICTION_KEY):
                logger.debug("Setting default prediction data")
                SessionState.set(cls.PREDICTION_KEY, None)
            
            # Initialize drawing state
            if not SessionState.has_key(cls.IS_DRAWING_KEY):
                logger.debug("Setting default drawing state")
                SessionState.set(cls.IS_DRAWING_KEY, False)
            
            # Initialize last change timestamp
            if not SessionState.has_key(cls.LAST_CHANGE_KEY):
                logger.debug("Setting default last change timestamp")
                SessionState.set(cls.LAST_CHANGE_KEY, time.time())
            
            # Initialize brush size
            if not SessionState.has_key(cls.BRUSH_SIZE_KEY):
                logger.debug(f"Setting default brush size: {cls.DEFAULT_BRUSH_SIZE}")
                SessionState.set(cls.BRUSH_SIZE_KEY, cls.DEFAULT_BRUSH_SIZE)
            
            # Initialize brush color
            if not SessionState.has_key(cls.BRUSH_COLOR_KEY):
                logger.debug(f"Setting default brush color: {cls.DEFAULT_BRUSH_COLOR}")
                SessionState.set(cls.BRUSH_COLOR_KEY, cls.DEFAULT_BRUSH_COLOR)
            
            # Initialize canvas dimensions
            if not SessionState.has_key(cls.CANVAS_WIDTH_KEY):
                logger.debug(f"Setting default canvas width: {cls.DEFAULT_CANVAS_WIDTH}")
                SessionState.set(cls.CANVAS_WIDTH_KEY, cls.DEFAULT_CANVAS_WIDTH)
            
            if not SessionState.has_key(cls.CANVAS_HEIGHT_KEY):
                logger.debug(f"Setting default canvas height: {cls.DEFAULT_CANVAS_HEIGHT}")
                SessionState.set(cls.CANVAS_HEIGHT_KEY, cls.DEFAULT_CANVAS_HEIGHT)
            
            # Initialize empty canvas data
            if not SessionState.has_key(cls.CANVAS_DATA_KEY):
                logger.debug("Setting empty canvas data")
                SessionState.set(cls.CANVAS_DATA_KEY, "")
            
            # Initialize current image
            if not SessionState.has_key(cls.CURRENT_IMAGE_KEY):
                logger.debug("Setting current image to None")
                SessionState.set(cls.CURRENT_IMAGE_KEY, None)
                
            logger.debug("Canvas state initialization complete")
        except Exception as e:
            logger.error(f"Error initializing canvas state: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_setting(cls, setting_name: str) -> Any:
        """Get a canvas setting.
        
        Args:
            setting_name: Name of the setting to retrieve.
            
        Returns:
            Any: The setting value.
        """
        logger.debug(f"Getting setting: {setting_name}")
        try:
            settings = SessionState.get(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            value = settings.get(setting_name, None)
            logger.debug(f"Retrieved setting: {setting_name}={value}")
            return value
        except Exception as e:
            logger.error(f"Error getting setting: {setting_name}, {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def update_setting(cls, setting_name: str, value: Any) -> None:
        """Update a canvas setting.
        
        Args:
            setting_name: Name of the setting to update.
            value: New value for the setting.
        """
        logger.debug(f"Updating setting: {setting_name} to {value}")
        try:
            settings = SessionState.get(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            settings[setting_name] = value
            SessionState.set(cls.SETTINGS_KEY, settings)
            logger.debug(f"Setting updated: {setting_name}={value}")
        except Exception as e:
            logger.error(f"Error updating setting: {setting_name}, {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def update_multiple_settings(cls, new_settings: Dict[str, Any]) -> None:
        """Update multiple canvas settings at once.
        
        Args:
            new_settings: Dictionary of settings to update.
        """
        logger.debug("Updating multiple settings")
        try:
            settings = SessionState.get(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            settings.update(new_settings)
            SessionState.set(cls.SETTINGS_KEY, settings)
            logger.debug("Settings updated")
        except Exception as e:
            logger.error(f"Error updating multiple settings: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all canvas settings.
        
        Returns:
            Dict[str, Any]: All canvas settings.
        """
        logger.debug("Getting all settings")
        try:
            settings = SessionState.get(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            logger.debug(f"Retrieved settings: {settings}")
            return settings
        except Exception as e:
            logger.error(f"Error getting all settings: {str(e)}", exc_info=True)
            return cls.DEFAULT_SETTINGS
    
    @classmethod
    def reset_settings(cls) -> None:
        """Reset canvas settings to defaults."""
        logger.debug("Resetting canvas settings")
        try:
            SessionState.set(cls.SETTINGS_KEY, cls.DEFAULT_SETTINGS)
            logger.debug("Canvas settings reset")
        except Exception as e:
            logger.error(f"Error resetting canvas settings: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def set_current_drawing(cls, drawing_data: Any) -> None:
        """Set the current drawing data.
        
        Args:
            drawing_data: The drawing data from the canvas.
        """
        logger.debug("Setting current drawing data")
        try:
            SessionState.set(cls.CURRENT_DRAWING_KEY, drawing_data)
            SessionState.set(cls.LAST_CHANGE_KEY, time.time())
            
            # Update drawing state
            if drawing_data:
                logger.debug("Drawing data provided, setting drawing state to True")
                SessionState.set(cls.IS_DRAWING_KEY, True)
        except Exception as e:
            logger.error(f"Error setting current drawing data: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_current_drawing(cls) -> Any:
        """Get the current drawing data.
        
        Returns:
            Any: The current drawing data.
        """
        logger.debug("Getting current drawing data")
        try:
            drawing_data = SessionState.get(cls.CURRENT_DRAWING_KEY)
            logger.debug(f"Retrieved current drawing data: {'None' if drawing_data is None else 'Not Empty'}")
            return drawing_data
        except Exception as e:
            logger.error(f"Error getting current drawing data: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def clear_drawing(cls) -> None:
        """Clear the current drawing."""
        logger.debug("Clearing current drawing")
        try:
            SessionState.set(cls.CURRENT_DRAWING_KEY, None)
            SessionState.set(cls.DRAWING_DATA_KEY, None)
            SessionState.set(cls.IS_DRAWING_KEY, False)
            SessionState.set(cls.LAST_CHANGE_KEY, time.time())
            logger.info("Current drawing cleared")
        except Exception as e:
            logger.error(f"Error clearing current drawing: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def set_drawing_data(cls, data: Any) -> None:
        """Set processed drawing data for prediction.
        
        Args:
            data: Processed drawing data (e.g., numpy array).
        """
        logger.debug("Setting drawing data for prediction")
        try:
            SessionState.set(cls.DRAWING_DATA_KEY, data)
            logger.debug("Drawing data set for prediction")
        except Exception as e:
            logger.error(f"Error setting drawing data for prediction: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_drawing_data(cls) -> Any:
        """Get processed drawing data.
        
        Returns:
            Any: Processed drawing data.
        """
        logger.debug("Getting drawing data for prediction")
        try:
            drawing_data = SessionState.get(cls.DRAWING_DATA_KEY)
            logger.debug(f"Retrieved drawing data: {'None' if drawing_data is None else 'Not Empty'}")
            return drawing_data
        except Exception as e:
            logger.error(f"Error getting drawing data for prediction: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def set_prediction(cls, prediction: Dict[str, Any]) -> None:
        """Set prediction results for the current drawing.
        
        Args:
            prediction: Prediction results (digit, confidence, etc.).
        """
        logger.debug("Setting prediction results")
        try:
            SessionState.set(cls.PREDICTION_KEY, prediction)
            logger.debug("Prediction results set")
        except Exception as e:
            logger.error(f"Error setting prediction results: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_prediction(cls) -> Optional[Dict[str, Any]]:
        """Get prediction results for the current drawing.
        
        Returns:
            Optional[Dict[str, Any]]: Prediction results or None.
        """
        logger.debug("Getting prediction results")
        try:
            prediction = SessionState.get(cls.PREDICTION_KEY)
            
            # Log the prediction in a safe way
            if prediction is None:
                logger.debug("Retrieved prediction results: None")
            else:
                digit = prediction.get("digit", "unknown")
                confidence = prediction.get("confidence", 0)
                logger.debug(f"Retrieved prediction results: Digit: {digit}, Confidence: {confidence}")
                
            return prediction
        except Exception as e:
            logger.error(f"Error getting prediction results: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def clear_prediction(cls) -> None:
        """Clear the current prediction."""
        logger.debug("Clearing current prediction")
        try:
            SessionState.set(cls.PREDICTION_KEY, None)
            logger.info("Current prediction cleared")
        except Exception as e:
            logger.error(f"Error clearing current prediction: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def is_drawing(cls) -> bool:
        """Check if a drawing is currently active.
        
        Returns:
            bool: True if a drawing exists, False otherwise.
        """
        logger.debug("Checking if drawing is active")
        try:
            drawing_state = SessionState.get(cls.IS_DRAWING_KEY, False)
            logger.debug(f"Retrieved drawing state: {drawing_state}")
            return drawing_state
        except Exception as e:
            logger.error(f"Error checking drawing state: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def get_last_change_time(cls) -> float:
        """Get the timestamp of the last drawing change.
        
        Returns:
            float: Timestamp of the last change.
        """
        logger.debug("Getting last change timestamp")
        try:
            last_change = SessionState.get(cls.LAST_CHANGE_KEY, 0)
            logger.debug(f"Retrieved last change timestamp: {last_change}")
            return last_change
        except Exception as e:
            logger.error(f"Error getting last change timestamp: {str(e)}", exc_info=True)
            return 0
    
    @classmethod
    def is_ready_for_prediction(cls) -> bool:
        """Check if the current drawing is ready for prediction.
        
        Returns:
            bool: True if the drawing is ready for prediction.
        """
        logger.debug("Checking if drawing is ready for prediction")
        try:
            return cls.is_drawing() and not cls.get_prediction()
        except Exception as e:
            logger.error(f"Error checking drawing readiness for prediction: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def get_brush_size(cls) -> int:
        """Get the current brush size.
        
        Returns:
            int: Current brush size
        """
        logger.debug("Getting brush size")
        try:
            brush_size = SessionState.get(cls.BRUSH_SIZE_KEY)
            logger.debug(f"Retrieved brush size: {brush_size}")
            return brush_size
        except Exception as e:
            logger.error(f"Error getting brush size: {str(e)}", exc_info=True)
            return cls.DEFAULT_BRUSH_SIZE
    
    @classmethod
    def set_brush_size(cls, size: int) -> None:
        """Set the brush size.
        
        Args:
            size: New brush size
        """
        logger.debug(f"Setting brush size to: {size}")
        try:
            SessionState.set(cls.BRUSH_SIZE_KEY, size)
            logger.debug(f"Brush size set to: {size}")
        except Exception as e:
            logger.error(f"Error setting brush size: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_brush_color(cls) -> str:
        """Get the current brush color.
        
        Returns:
            str: Current brush color
        """
        logger.debug("Getting brush color")
        try:
            brush_color = SessionState.get(cls.BRUSH_COLOR_KEY)
            logger.debug(f"Retrieved brush color: {brush_color}")
            return brush_color
        except Exception as e:
            logger.error(f"Error getting brush color: {str(e)}", exc_info=True)
            return cls.DEFAULT_BRUSH_COLOR
    
    @classmethod
    def set_brush_color(cls, color: str) -> None:
        """Set the brush color.
        
        Args:
            color: New brush color
        """
        logger.debug(f"Setting brush color to: {color}")
        try:
            SessionState.set(cls.BRUSH_COLOR_KEY, color)
            logger.debug(f"Brush color set to: {color}")
        except Exception as e:
            logger.error(f"Error setting brush color: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_canvas_dimensions(cls) -> Tuple[int, int]:
        """Get the canvas dimensions.
        
        Returns:
            Tuple[int, int]: Canvas width and height
        """
        logger.debug("Getting canvas dimensions")
        try:
            width = SessionState.get(cls.CANVAS_WIDTH_KEY)
            height = SessionState.get(cls.CANVAS_HEIGHT_KEY)
            logger.debug(f"Retrieved canvas dimensions: {width}x{height}")
            return width, height
        except Exception as e:
            logger.error(f"Error getting canvas dimensions: {str(e)}", exc_info=True)
            return cls.DEFAULT_CANVAS_WIDTH, cls.DEFAULT_CANVAS_HEIGHT
    
    @classmethod
    def set_canvas_dimensions(cls, width: int, height: int) -> None:
        """Set the canvas dimensions.
        
        Args:
            width: Canvas width
            height: Canvas height
        """
        logger.debug(f"Setting canvas dimensions to: {width}x{height}")
        try:
            SessionState.set(cls.CANVAS_WIDTH_KEY, width)
            SessionState.set(cls.CANVAS_HEIGHT_KEY, height)
            logger.debug(f"Canvas dimensions set to: {width}x{height}")
        except Exception as e:
            logger.error(f"Error setting canvas dimensions: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_canvas_data(cls) -> str:
        """Get the canvas data (as base64 string).
        
        Returns:
            str: Canvas data as base64 string
        """
        logger.debug("Getting canvas data")
        try:
            data = SessionState.get(cls.CANVAS_DATA_KEY, "")
            logger.debug(f"Retrieved canvas data: {'Empty' if not data else data[:20] + '...'}")
            return data
        except Exception as e:
            logger.error(f"Error getting canvas data: {str(e)}", exc_info=True)
            return ""
    
    @classmethod
    def set_canvas_data(cls, data: str) -> None:
        """Set the canvas data.
        
        Args:
            data: Canvas data as base64 string
        """
        logger.debug("Setting canvas data")
        try:
            SessionState.set(cls.CANVAS_DATA_KEY, data)
            
            # Also update current image if data is provided
            if data:
                logger.debug("Canvas data provided, updating current image")
                # Convert to numpy array for processing
                image = cls._data_uri_to_image(data)
                SessionState.set(cls.CURRENT_IMAGE_KEY, image)
                logger.debug("Current image updated from canvas data")
            else:
                logger.debug("Clearing canvas data and current image")
                SessionState.set(cls.CURRENT_IMAGE_KEY, None)
        except Exception as e:
            logger.error(f"Error setting canvas data: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def clear_canvas(cls) -> None:
        """Clear the canvas data."""
        logger.debug("Clearing canvas")
        try:
            SessionState.set(cls.CANVAS_DATA_KEY, "")
            SessionState.set(cls.CURRENT_IMAGE_KEY, None)
            logger.info("Canvas cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing canvas: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_current_image(cls) -> Optional[np.ndarray]:
        """Get the current image as numpy array.
        
        Returns:
            Optional[np.ndarray]: Current image or None if no image
        """
        logger.debug("Getting current image")
        try:
            image = SessionState.get(cls.CURRENT_IMAGE_KEY)
            logger.debug(f"Retrieved current image: {'None' if image is None else f'Array shape: {image.shape}'}")
            return image
        except Exception as e:
            logger.error(f"Error getting current image: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def set_current_image(cls, image: Optional[np.ndarray]) -> None:
        """Set the current image.
        
        Args:
            image: Image as numpy array or None to clear
        """
        logger.debug(f"Setting current image: {'None' if image is None else f'Array shape: {image.shape}'}")
        try:
            SessionState.set(cls.CURRENT_IMAGE_KEY, image)
            
            # Clear canvas data since we're setting image directly
            if image is not None:
                logger.debug("Clearing canvas data since image is being set directly")
                SessionState.set(cls.CANVAS_DATA_KEY, "")
                
            logger.debug("Current image set successfully")
        except Exception as e:
            logger.error(f"Error setting current image: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def _data_uri_to_image(cls, data_uri: str) -> np.ndarray:
        """Convert data URI to numpy image.
        
        Args:
            data_uri: Data URI string
            
        Returns:
            np.ndarray: Image as numpy array
        """
        logger.debug("Converting data URI to image")
        try:
            # Extract base64 data from URI
            base64_data = data_uri.split(',')[1]
            
            # Convert to PIL Image
            image_data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_data))
            
            # Convert to numpy array
            array = np.array(image)
            
            logger.debug(f"Converted data URI to image with shape: {array.shape}")
            return array
        except Exception as e:
            logger.error(f"Error converting data URI to image: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def _image_to_data_uri(cls, image: np.ndarray, format: str = 'PNG') -> str:
        """Convert numpy image to data URI.
        
        Args:
            image: Image as numpy array
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            str: Data URI string
        """
        logger.debug(f"Converting image to data URI with format: {format}")
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to BytesIO
            buffer = BytesIO()
            pil_image.save(buffer, format=format)
            
            # Convert to base64
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Create data URI
            data_uri = f"data:image/{format.lower()};base64,{base64_data}"
            
            logger.debug(f"Converted image to data URI, length: {len(data_uri)}")
            return data_uri
        except Exception as e:
            logger.error(f"Error converting image to data URI: {str(e)}", exc_info=True)
            raise 