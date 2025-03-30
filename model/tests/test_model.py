# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/tests/test_model.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-30
# Updated: 2025-03-30

import pytest
import torch
from model import MNISTClassifier


def test_model_architecture():
    """Test that the model architecture is as expected."""
    model = MNISTClassifier()

    # Check that we have the expected layers
    assert hasattr(model, "conv1")
    assert hasattr(model, "conv2")
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, 10)


def test_model_predict():
    """Test the model's predict function."""
    model = MNISTClassifier()

    # Create a dummy input
    batch_size = 5
    x = torch.randn(batch_size, 1, 28, 28)

    # Test prediction
    predicted_class, confidence = model.predict(x)

    # Check that predicted_class is of the right shape
    assert predicted_class.shape == (batch_size,)

    # Check that confidence is of the right shape
    assert confidence.shape == (batch_size,)

    # Check that confidence values are between 0 and 1
    assert torch.all(confidence >= 0)
    assert torch.all(confidence <= 1)


if __name__ == "__main__":
    test_model_architecture()
    test_model_predict()
    print("All tests passed!")
