# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
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
from utils.aspects import AspectUtils

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

    _logger = logging.getLogger(f"{__name__}.CanvasState")

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize canvas state with default values."""
        if SessionState.get(cls.CANVAS_KEY) is None:
            SessionState.set(
                cls.CANVAS_KEY,
                {"strokes": [], "is_empty": True, "last_updated": None},
            )

        if SessionState.get(cls.INPUT_TYPE_KEY) is None:
            SessionState.set(cls.INPUT_TYPE_KEY, cls.CANVAS_INPUT)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_input_type(cls) -> str:
        """Get current input type.

        Returns:
            Current input type (canvas, upload, url)
        """
        cls.initialize()
        return SessionState.get(cls.INPUT_TYPE_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_input_type(cls, input_type: str) -> None:
        """Set input type.

        Args:
            input_type: Input type to set (canvas, upload, url)
        """
        if input_type not in [
            cls.CANVAS_INPUT,
            cls.UPLOAD_INPUT,
            cls.URL_INPUT,
        ]:
            logger.warning(f"Invalid input type: {input_type}")
            return

        cls.initialize()
        SessionState.set(cls.INPUT_TYPE_KEY, input_type)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_canvas_data(cls) -> Dict[str, Any]:
        """Get current canvas data.

        Returns:
            Dictionary with canvas stroke data
        """
        cls.initialize()
        return SessionState.get(cls.CANVAS_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
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
            "last_updated": datetime.now().isoformat(),
        }
        SessionState.set(cls.CANVAS_KEY, canvas_data)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def clear_canvas(cls) -> None:
        """Clear canvas data."""
        cls.initialize()
        canvas_data = {"strokes": [], "is_empty": True, "last_updated": None}
        SessionState.set(cls.CANVAS_KEY, canvas_data)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_image_data(
        cls, image_data: Union[bytes, np.ndarray, Image.Image]
    ) -> None:
        """Set image data with enhanced error handling and verification.

        Args:
            image_data: Image data as bytes, numpy array, or PIL Image
        """
        cls.initialize()

        try:
            # Convert to bytes if needed
            if isinstance(image_data, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(image_data.astype("uint8"))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                cls._logger.debug("Converted numpy array to bytes")
            elif isinstance(image_data, Image.Image):
                buffer = io.BytesIO()
                image_data.save(buffer, format="PNG")
                image_data = buffer.getvalue()
                cls._logger.debug("Converted PIL Image to bytes")

            # Verify it's valid image data
            if not isinstance(image_data, bytes):
                raise ValueError(
                    f"Invalid image data type: {type(image_data)}"
                )

            # Check if data is empty
            if len(image_data) == 0:
                raise ValueError("Empty image data")

            # Try to open as image to verify it's valid
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # Verify it's a valid image
                cls._logger.debug(
                    f"Valid image data verified: {img.format} {img.size[0]}x{img.size[1]}"
                )
            except Exception as e:
                cls._logger.warning(f"Image verification failed: {str(e)}")
                # Continue anyway as some raw data might still be usable by the model

            # Store as base64 string
            encoded = base64.b64encode(image_data).decode("utf-8")
            SessionState.set(cls.IMAGE_DATA_KEY, encoded)
            cls._logger.info(
                f"Image data stored: {len(encoded)} chars (base64)"
            )

            # Store timestamp for debugging
            import time

            SessionState.set("last_image_update", time.time())
        except Exception as e:
            cls._logger.error(
                f"Error setting image data: {str(e)}", exc_info=True
            )
            raise

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_image_data(cls) -> Optional[bytes]:
        """Get image data with verification.

        Returns:
            Image data as bytes or None if not set
        """
        cls.initialize()
        encoded = SessionState.get(cls.IMAGE_DATA_KEY)
        if encoded:
            try:
                decoded = base64.b64decode(encoded)
                cls._logger.debug(
                    f"Retrieved image data: {len(decoded)} bytes"
                )
                return decoded
            except Exception as e:
                cls._logger.error(
                    f"Error decoding image data: {str(e)}", exc_info=True
                )
                return None
        else:
            cls._logger.debug("No image data found in session state")
            return None

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def check_image_data(cls) -> Tuple[bool, Optional[str]]:
        """
        Check if image data exists and is valid.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            data = cls.get_image_data()
            if data is None:
                return False, "No image data available"

            if len(data) == 0:
                return False, "Empty image data"

            # Try to open the image to verify it's valid
            img = Image.open(io.BytesIO(data))
            img.verify()

            return True, None
        except Exception as e:
            return False, str(e)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_processed_image_data(cls) -> Optional[bytes]:
        """Get image data processed for the model.

        This applies any necessary preprocessing before sending to model API.

        Returns:
            Processed image data as bytes or None if not available
        """
        # Get raw image data
        image_data = cls.get_image_data()
        if not image_data:
            return None

        try:
            # Open as PIL Image
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_data)).convert("L")

            # Basic preprocessing - center the digit
            img = cls._center_digit(img)

            # Convert back to bytes
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        except Exception as e:
            cls._logger.error(f"Error processing image: {str(e)}")
            return image_data  # Return original if processing fails

    @classmethod
    def _center_digit(cls, image: "Image.Image") -> "Image.Image":
        """Center the digit in the image.

        Args:
            image: PIL Image

        Returns:
            Centered PIL Image
        """
        try:
            import numpy as np

            # Convert to numpy array
            img_array = np.array(image)

            # Invert if white digit on black background
            if np.mean(img_array) > 127:
                img_array = 255 - img_array

            # Find non-zero pixels (the digit)
            rows, cols = np.where(img_array < 127)

            # If no digit found, return original
            if len(rows) == 0 or len(cols) == 0:
                return image

            # Find bounding box
            top, bottom = np.min(rows), np.max(rows)
            left, right = np.min(cols), np.max(cols)

            # Calculate center of digit
            center_y, center_x = (top + bottom) // 2, (left + right) // 2

            # Calculate center of image
            height, width = img_array.shape
            img_center_y, img_center_x = height // 2, width // 2

            # Calculate translation
            dy, dx = img_center_y - center_y, img_center_x - center_x

            # Create new image and paste with offset
            from PIL import Image

            new_img = Image.new("L", (width, height), 255)
            new_img.paste(image, (dx, dy))

            return new_img
        except Exception as e:
            cls._logger.error(f"Error centering digit: {str(e)}")
            return image  # Return original if centering fails

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_image_url(cls, url: str) -> None:
        """Set image URL.

        Args:
            url: URL to image
        """
        cls.initialize()
        SessionState.set(cls.URL_KEY, url)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_image_url(cls) -> Optional[str]:
        """Get image URL.

        Returns:
            Image URL or None if not set
        """
        cls.initialize()
        return SessionState.get(cls.URL_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def clear_all(cls) -> None:
        """Clear all image data (canvas, upload, URL)."""
        cls.clear_canvas()
        SessionState.set(cls.IMAGE_DATA_KEY, None)
        SessionState.set(cls.URL_KEY, None)
