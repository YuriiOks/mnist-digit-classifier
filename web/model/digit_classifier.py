# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/model/digit_classifier.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-07
# Updated: 2025-03-30

import numpy as np
import requests
import base64
import io
import logging
import os
import time
from PIL import Image
from typing import Tuple, Optional, Dict, Any, Union


class DigitClassifier:
    """Model client for classifying digits.

    This class handles communication with the model API service,
    including preprocessing and error handling.
    """

    # Get model API URL from environment or use default for Docker
    MODEL_API_URL = os.environ.get("MODEL_URL", "http://model:5000")

    # Prediction endpoint
    PREDICT_ENDPOINT = "/predict"

    # Health check endpoint
    HEALTH_ENDPOINT = "/health"

    def __init__(self):
        """Initialize the digit classifier client."""
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initializing DigitClassifier with API URL: {self.MODEL_API_URL}"
        )

    def check_health(self) -> bool:
        """Check if the model API is available.

        Returns:
            bool: True if the model API is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.MODEL_API_URL}{self.HEALTH_ENDPOINT}", timeout=2
            )
            is_healthy = response.status_code == 200
            if is_healthy:
                self.logger.info("Model API health check: HEALTHY")
            else:
                self.logger.warning(
                    f"Model API returned non-200 status: {response.status_code}"
                )
            return is_healthy
        except requests.RequestException as e:
            self.logger.warning(f"Model API health check failed: {str(e)}")
            return False

    @staticmethod
    def preprocess_image(
        image_data: Union[bytes, Image.Image, np.ndarray],
    ) -> str:
        """Preprocess image data for the model API.

        Args:
            image_data: Raw image data from canvas or upload

        Returns:
            Base64 encoded image data ready for API
        """
        logger = logging.getLogger(__name__)

        try:
            # Convert image to bytes if it's not already
            if isinstance(image_data, Image.Image):
                # Convert PIL Image to bytes
                logger.debug("Converting PIL Image to bytes")
                img_bytes = io.BytesIO()
                image_data.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()
            elif isinstance(image_data, np.ndarray):
                # Convert numpy array to PIL Image to bytes
                logger.debug("Converting numpy array to bytes")
                img = Image.fromarray(image_data.astype("uint8"))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()
            else:
                # Assume it's already bytes
                logger.debug(f"Using raw bytes ({len(image_data)} bytes)")
                img_bytes = image_data

            # Encode to base64
            base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
            logger.debug(f"Image encoded to base64 ({len(base64_encoded)} characters)")
            return base64_encoded
        except Exception as e:
            logger.error(f"Error in preprocess_image: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def predict(
        self, image_data: Union[bytes, Image.Image, np.ndarray]
    ) -> Tuple[int, float]:
        """Predict the digit in the image.

        Args:
            image_data: Raw image data from canvas, upload, or URL

        Returns:
            Tuple of (predicted_digit, confidence)

        Raises:
            ConnectionError: If the model API is unavailable
            ValueError: If the image processing fails
        """
        start_time = time.time()
        self.logger.info("Starting prediction request")

        try:
            # Check if API is healthy
            if not self.check_health():
                self.logger.error("Model API is not available")
                raise ConnectionError(
                    "Model service is not available. Please try again later."
                )

            # Preprocess image
            base64_image = self.preprocess_image(image_data)
            self.logger.debug(
                f"Image preprocessed successfully, base64 length: {len(base64_image)}"
            )

            # Prepare request payload
            payload = {"image": base64_image}

            # Send prediction request
            self.logger.debug("Sending prediction request to model API")
            response = requests.post(
                f"{self.MODEL_API_URL}{self.PREDICT_ENDPOINT}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,  # Increased timeout for larger images
            )

            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                predicted_digit = int(result["prediction"])
                confidence = float(result["confidence"])

                duration = time.time() - start_time
                self.logger.info(
                    f"Prediction successful in {duration:.2f}s: digit={predicted_digit}, confidence={confidence:.2f}"
                )
                return predicted_digit, confidence
            else:
                error_msg = f"Model API returned error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)

                try:
                    # Try to parse error message from JSON response
                    error_json = response.json()
                    if "error" in error_json:
                        error_msg = f"Model API error: {error_json['error']}"
                except:
                    pass

                raise ValueError(error_msg)

        except requests.exceptions.Timeout:
            error_msg = (
                "Request to model API timed out. The server might be overloaded."
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error communicating with model API: {str(e)}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during prediction: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)

    def batch_predict(self, images: list) -> list:
        """Predict digits for multiple images.

        Args:
            images: List of image data

        Returns:
            List of (digit, confidence) tuples
        """
        results = []
        for i, image in enumerate(images):
            try:
                self.logger.info(f"Processing batch image {i+1}/{len(images)}")
                digit, confidence = self.predict(image)
                results.append((digit, confidence))
            except Exception as e:
                self.logger.error(f"Error predicting image {i+1} in batch: {str(e)}")
                results.append((None, 0.0))

        return results
