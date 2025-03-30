# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/inputs/file_upload.py
# Description: Image file upload component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, List
import io
from PIL import Image
import numpy as np

from ui.components.base.component import Component
from core.app_state.canvas_state import CanvasState

logger = logging.getLogger(__name__)


class ImageUpload(Component):
    """Image upload component for digit classification."""

    def __init__(
        self,
        key: str = "image_upload",
        on_upload: Optional[Callable] = None,
        classes: List[str] = None,
        attributes: Dict[str, str] = None,
    ):
        """Initialize image upload component.

        Args:
            key: Unique key for the component
            on_upload: Callback for successful upload
            classes: Additional CSS classes
            attributes: Additional HTML attributes
        """
        super().__init__(classes=classes, attributes=attributes)
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.key = key
        self.on_upload = on_upload

    def render(self) -> None:
        """Render the image upload component."""
        self.logger.debug("Rendering image upload")

        try:
            # Custom CSS for file uploader
            st.markdown(
                """
            <style>
            .uploadfile-wrapper {
                margin-bottom: 1rem;
            }
            
            .uploadfile-label {
                font-weight: 500;
                margin-bottom: 0.5rem;
                display: block;
            }
            
            .uploadfile-hint {
                font-size: 0.875rem;
                color: #666;
                margin-top: 0.5rem;
            }
            
            .stButton button {
                width: 100%;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # File uploader
            st.markdown(
                '<div class="uploadfile-label">Upload an image of a digit</div>',
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Upload an image of a handwritten digit",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key=self.key,
            )

            st.markdown(
                '<div class="uploadfile-hint">Supported formats: PNG, JPG, JPEG</div>',
                unsafe_allow_html=True,
            )

            # Process uploaded file
            if uploaded_file is not None:
                try:
                    # Read image
                    img_bytes = uploaded_file.getvalue()

                    # Save to state
                    CanvasState.set_image_data(img_bytes)
                    CanvasState.set_input_type(CanvasState.UPLOAD_INPUT)

                    # Display preview
                    img = Image.open(io.BytesIO(img_bytes))
                    img_resized = self._resize_image(img, (150, 150))
                    st.image(
                        img_resized,
                        caption="Uploaded Image",
                        use_column_width=False,
                    )

                    # Call callback if provided
                    if self.on_upload:
                        self.on_upload()

                    self.logger.debug(
                        f"Successfully processed uploaded image: {uploaded_file.name}"
                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    self.logger.error(
                        f"Error processing uploaded image: {str(e)}",
                        exc_info=True,
                    )

        except Exception as e:
            self.logger.error(
                f"Error rendering image upload: {str(e)}", exc_info=True
            )
            st.error(
                f"An error occurred while rendering the image upload: {str(e)}"
            )

    def _resize_image(self, img: Image.Image, size: tuple) -> Image.Image:
        """Resize image while preserving aspect ratio.

        Args:
            img: PIL Image object
            size: Target size (width, height)

        Returns:
            Resized PIL Image
        """
        # Calculate aspect ratio
        width, height = img.size
        target_width, target_height = size

        # Determine new dimensions
        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Create blank image with target size
        img_with_padding = Image.new("RGB", size, color="white")

        # Paste resized image in center
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        img_with_padding.paste(img_resized, (paste_x, paste_y))

        return img_with_padding
