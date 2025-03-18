import numpy as np
import requests
import base64
import io
import logging
import os
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
        self.logger.info(f"Initializing DigitClassifier with API URL: {self.MODEL_API_URL}")
    
    def check_health(self) -> bool:
        """Check if the model API is available.
        
        Returns:
            bool: True if the model API is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.MODEL_API_URL}{self.HEALTH_ENDPOINT}", 
                timeout=2
            )
            return response.status_code == 200
        except requests.RequestException as e:
            self.logger.warning(f"Model API health check failed: {str(e)}")
            return False
    
    @staticmethod
    def preprocess_image(image_data: Union[bytes, Image.Image, np.ndarray]) -> bytes:
        """Preprocess image data for the model API.
        
        Args:
            image_data: Raw image data from canvas or upload
            
        Returns:
            Base64 encoded image data ready for API
        """
        # Convert image to bytes if it's not already
        if isinstance(image_data, Image.Image):
            # Convert PIL Image to bytes
            img_bytes = io.BytesIO()
            image_data.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
        elif isinstance(image_data, np.ndarray):
            # Convert numpy array to PIL Image to bytes
            img = Image.fromarray(image_data.astype('uint8'))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
        else:
            # Assume it's already bytes
            img_bytes = image_data
        
        # Encode to base64
        base64_encoded = base64.b64encode(img_bytes).decode('utf-8')
        return base64_encoded
    
    def predict(self, image_data: Union[bytes, Image.Image, np.ndarray]) -> Tuple[int, float]:
        """Predict the digit in the image.
        
        Args:
            image_data: Raw image data from canvas, upload, or URL
            
        Returns:
            Tuple of (predicted_digit, confidence)
            
        Raises:
            ConnectionError: If the model API is unavailable
            ValueError: If the image processing fails
        """
        try:
            # Check if API is healthy
            if not self.check_health():
                self.logger.error("Model API is not available")
                raise ConnectionError("Model service is not available. Please try again later.")
            
            # Preprocess image
            base64_image = self.preprocess_image(image_data)
            
            # Prepare request payload
            payload = {"image": base64_image}
            
            # Send prediction request
            self.logger.debug("Sending prediction request to model API")
            response = requests.post(
                f"{self.MODEL_API_URL}{self.PREDICT_ENDPOINT}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                predicted_digit = int(result["prediction"])
                confidence = float(result["confidence"])
                self.logger.info(f"Prediction successful: digit={predicted_digit}, confidence={confidence:.2f}")
                return predicted_digit, confidence
            else:
                error_msg = f"Model API returned error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
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
        for image in images:
            try:
                digit, confidence = self.predict(image)
                results.append((digit, confidence))
            except Exception as e:
                self.logger.error(f"Error predicting image in batch: {str(e)}")
                results.append((None, 0.0))
        
        return results