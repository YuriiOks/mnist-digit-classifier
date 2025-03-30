# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/tests/test_inference.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-06
# Updated: 2025-03-30

import pytest
import torch
import numpy as np
from PIL import Image
import os
import tempfile
from inference import MNISTPredictor

# Skip tests if model file doesn't exist
pytestmark = pytest.mark.skipif(
    not os.path.exists("saved_models/mnist_classifier.pt"),
    reason="Trained model not found",
)


def create_dummy_model():
    """Create a dummy model file for testing."""
    from model import MNISTClassifier

    # Create a model
    model = MNISTClassifier()

    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    # Save model
    torch.save(model.state_dict(), "saved_models/mnist_classifier.pt")


def test_predictor_initialization():
    """Test that the predictor initializes correctly."""
    # Create dummy model if real one doesn't exist
    if not os.path.exists("saved_models/mnist_classifier.pt"):
        create_dummy_model()

    # Initialize predictor
    predictor = MNISTPredictor()

    # Check that model is in eval mode
    assert not predictor.model.training


def test_prediction():
    """Test prediction functionality."""
    # Create dummy model if real one doesn't exist
    if not os.path.exists("saved_models/mnist_classifier.pt"):
        create_dummy_model()

    # Initialize predictor
    predictor = MNISTPredictor()

    # Create a dummy image
    image = Image.new("L", (28, 28), 255)

    # Draw a simple digit (e.g., a vertical line like '1')
    for i in range(5, 20):
        for j in range(10, 15):
            image.putpixel((j, i), 0)

    # Make prediction
    digit, confidence = predictor.predict(image)

    # Check that digit is an integer
    assert isinstance(digit, int)

    # Check that confidence is a float between 0 and 1
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1


def test_batch_prediction():
    """Test batch prediction functionality."""
    # Create dummy model if real one doesn't exist
    if not os.path.exists("saved_models/mnist_classifier.pt"):
        create_dummy_model()

    # Initialize predictor
    predictor = MNISTPredictor()

    # Create dummy images
    images = []
    for _ in range(3):
        image = Image.new("L", (28, 28), 255)
        # Draw a simple digit
        for i in range(5, 20):
            for j in range(10, 15):
                image.putpixel((j, i), 0)
        images.append(image)

    # Make batch prediction
    digits, confidences = predictor.predict_batch(images)

    # Check results
    assert len(digits) == 3
    assert len(confidences) == 3

    # Check types
    for digit in digits:
        assert isinstance(digit, int)

    for confidence in confidences:
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    # Create dummy model for testing
    create_dummy_model()

    test_predictor_initialization()
    test_prediction()
    test_batch_prediction()
    print("All tests passed!")
