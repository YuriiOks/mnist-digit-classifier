# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/components/inputs/url_input.py
# Description: URL input component for image loading
# Created: 2024-05-01
# Updated: 2025-03-30

import streamlit as st
import logging
from typing import Dict, Any, Callable, Optional, List
import requests
from PIL import Image
from io import BytesIO
import re

from ui.components.base.component import Component
from core.app_state.canvas_state import CanvasState
from ui.components.controls.buttons import PrimaryButton

logger = logging.getLogger(__name__)


class ImageUrlInput(Component):
    """Image URL input component for digit classification."""

    def __init__(
        self,
        key: str = "image_url",
        on_load: Optional[Callable] = None,
        classes: List[str] = None,
        attributes: Dict[str, str] = None,
    ):
        """Initialize image URL input component.

        Args:
            key: Unique key for the component
            on_load: Callback for successful image load
            classes: Additional CSS classes
            attributes: Additional HTML attributes
        """
        super().__init__(classes=classes, attributes=attributes)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.key = key
        self.on_load = on_load

    def render(self) -> None:
        """Render the image URL input component."""
        self.logger.debug("Rendering image URL input")

        try:
            # Custom CSS
            st.markdown(
                """
            <style>
            .urlinput-wrapper {
                margin-bottom: 1rem;
            }
            
            .urlinput-label {
                font-weight: 500;
                margin-bottom: 0.5rem;
                display: block;
            }
            
            .urlinput-hint {
                font-size: 0.875rem;
                color: #666;
                margin-top: 0.5rem;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # URL input field
            st.markdown(
                '<div class="urlinput-label">Enter image URL</div>',
                unsafe_allow_html=True,
            )

            # Get previously entered URL if any
            previous_url = CanvasState.get_image_url() or ""

            url = st.text_input(
                "Image URL",
                value=previous_url,
                placeholder="https://example.com/digit.jpg",
                label_visibility="collapsed",
                key=self.key,
            )

            st.markdown(
                '<div class="urlinput-hint">Enter URL to an image of a handwritten digit</div>',
                unsafe_allow_html=True,
            )

            # Load button
            if url:
                # Save URL to state
                if url != previous_url:
                    CanvasState.set_image_url(url)

                # Create load button
                load_btn = PrimaryButton(
                    label="Load Image",
                    icon="🔄",
                    on_click=lambda: self._load_image_from_url(url),
                )
                load_btn.display()

            # Display preview if already loaded
            if (
                CanvasState.get_input_type() == CanvasState.URL_INPUT
                and CanvasState.get_image_data()
            ):
                img_bytes = CanvasState.get_image_data()
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes))
                    img_resized = self._resize_image(img, (150, 150))
                    st.image(
                        img_resized,
                        caption="Loaded Image",
                        use_column_width=False,
                    )

        except Exception as e:
            self.logger.error(
                f"Error rendering image URL input: {str(e)}", exc_info=True
            )
            st.error(f"An error occurred while rendering the image URL input: {str(e)}")

    def _load_image_from_url(self, url: str) -> None:
        """Load image from URL and store in application state.

        Args:
            url: URL to image
        """
        self.logger.debug(f"Loading image from URL: {url}")

        # Basic URL validation
        if not self._is_valid_url(url):
            st.error("Invalid URL format. Please enter a valid image URL.")
            return

        try:
            # Fetch image from URL
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    st.error("The URL does not point to an image.")
                    return

                # Process image
                img_bytes = response.content
                CanvasState.set_image_data(img_bytes)
                CanvasState.set_input_type(CanvasState.URL_INPUT)

                # Call callback if provided
                if self.on_load:
                    self.on_load()

                self.logger.debug("Successfully loaded image from URL")
                st.success("Image loaded successfully!")
                st.rerun()
            else:
                st.error(f"Failed to load image. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image: {str(e)}")
            self.logger.error(f"Error fetching image from URL: {str(e)}", exc_info=True)

    def _is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        pattern = re.compile(
            r"^(?:http|https)://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(pattern.match(url))

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
