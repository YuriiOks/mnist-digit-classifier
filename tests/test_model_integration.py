# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: tests/test_model_integration.py
# Description: Test model API integration
# Created: 2025-03-17
# Updated: 2025-03-30

import unittest
import os
import sys
import requests
from PIL import Image
import io
import base64
import logging

# Add web directory to path for importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the digit classifier
from model.digit_classifier import DigitClassifier


class TestModelIntegration(unittest.TestCase):
    """Test integration with the model API."""

    @classmethod
    def setUpClass(cls):
        """Set up test class with test image."""
        # Create a simple test image (a vertical line like "1")
        cls.test_image = Image.new("L", (28, 28), 255)  # White background
        for i in range(5, 23):
            for j in range(12, 16):
                cls.test_image.putpixel((j, i), 0)  # Black pixels for the digit

        # Convert to bytes
        buffer = io.BytesIO()
        cls.test_image.save(buffer, format="PNG")
        cls.test_image_bytes = buffer.getvalue()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

        # Keep track of skip reason if model API is unavailable
        cls.skip_reason = None

        # Check if model API is available
        try:
            # Override MODEL_API_URL for testing if specified in environment
            cls.model_api_url = os.environ.get("TEST_MODEL_URL", "http://model:5000")
            response = requests.get(f"{cls.model_api_url}/health", timeout=2)
            if response.status_code != 200:
                cls.skip_reason = f"Model API returned status {response.status_code}"
        except requests.RequestException as e:
            cls.skip_reason = f"Model API is not available: {str(e)}"

    def setUp(self):
        """Skip tests if model API is not available."""
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_health_check(self):
        """Test health check functionality."""
        classifier = DigitClassifier()
        result = classifier.check_health()
        self.assertTrue(
            result,
            "Health check should return True if model API is available",
        )

    def test_image_preprocessing(self):
        """Test image preprocessing."""
        # Test with PIL Image
        encoded = DigitClassifier.preprocess_image(self.test_image)
        self.assertIsInstance(
            encoded, str, "Preprocessing PIL Image should return a string"
        )

        # Test with bytes
        encoded = DigitClassifier.preprocess_image(self.test_image_bytes)
        self.assertIsInstance(
            encoded, str, "Preprocessing bytes should return a string"
        )

        # Test with numpy array
        import numpy as np

        array = np.array(self.test_image)
        encoded = DigitClassifier.preprocess_image(array)
        self.assertIsInstance(
            encoded, str, "Preprocessing numpy array should return a string"
        )

    def test_prediction(self):
        """Test end-to-end prediction."""
        classifier = DigitClassifier()
        digit, confidence = classifier.predict(self.test_image)

        # Basic validation of return values
        self.assertIsInstance(digit, int, "Predicted digit should be an integer")
        self.assertIsInstance(confidence, float, "Confidence should be a float")
        self.assertTrue(0 <= confidence <= 1, "Confidence should be between 0 and 1")
        self.assertTrue(0 <= digit <= 9, "Digit should be between 0 and 9")

        self.logger.info(f"Predicted digit: {digit}, confidence: {confidence:.4f}")


if __name__ == "__main__":
    unittest.main()
