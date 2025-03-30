# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/model.py
# Description: Defines the CNN architecture for MNIST classification.
# Created: Earlier Date
# Updated: 2025-03-28 (Added BatchNorm layers to match saved model)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    CNN model for MNIST digit classification.

    Architecture includes Convolutional layers, Batch Normalization,
    Max Pooling, Dropout, and Fully Connected layers.
    """

    def __init__(self):
        """Initializes the layers of the CNN."""
        super(MNISTClassifier, self).__init__()

        print(
            "!!! Initializing MNISTClassifier with BatchNorm Layers (Version 2) !!!"
        )

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1,  # Input channels (grayscale image)
            out_channels=32,  # Number of filters
            kernel_size=3,  # Filter size
            stride=1,
            padding=1,  # Keep spatial dimension same (28x28)
        )
        # --- Add BatchNorm layer after conv1 ---
        self.bn1 = nn.BatchNorm2d(num_features=32)  # Match out_channels

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,  # Input channels from previous layer
            out_channels=64,  # Number of filters
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # --- Add BatchNorm layer after conv2 ---
        self.bn2 = nn.BatchNorm2d(num_features=64)  # Match out_channels

        # Dropout Layers
        # Dropout2d for feature maps (after conv/pool)
        self.dropout1 = nn.Dropout2d(p=0.25)
        # Regular Dropout for flattened features (in FC layers)
        self.dropout2 = nn.Dropout(p=0.5)

        # Fully Connected Layers
        # Input features: 64 channels * 7 height * 7 width
        # Calculation: 28x28 -> Pool(2) -> 14x14 -> Pool(2) -> 7x7
        fc1_input_features = 64 * 7 * 7  # Should be 3136
        self.fc1 = nn.Linear(in_features=fc1_input_features, out_features=128)
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10,  # 10 output classes for digits 0-9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor of raw logits
                          (batch_size, 10).
        """
        # Input shape: (batch_size, 1, 28, 28)

        # Conv Block 1
        x = self.conv1(x)  # -> (batch, 32, 28, 28)
        x = self.bn1(x)  # -> (batch, 32, 28, 28) Apply BatchNorm
        x = F.relu(x)  # -> (batch, 32, 28, 28) Apply activation
        x = F.max_pool2d(x, 2)  # -> (batch, 32, 14, 14) Pool 1

        # Conv Block 2
        x = self.conv2(x)  # -> (batch, 64, 14, 14)
        x = self.bn2(x)  # -> (batch, 64, 14, 14) Apply BatchNorm
        x = F.relu(x)  # -> (batch, 64, 14, 14) Apply activation
        x = F.max_pool2d(x, 2)  # -> (batch, 64, 7, 7)  Pool 2

        # Dropout on feature map
        x = self.dropout1(x)  # -> (batch, 64, 7, 7)

        # Flatten feature maps for Fully Connected layers
        # Start flattening from dimension 1 (keep batch dimension 0)
        x = torch.flatten(x, 1)  # -> (batch, 64*7*7) = (batch, 3136)

        # Fully Connected Block
        x = self.fc1(x)  # -> (batch, 128)
        x = F.relu(x)  # Apply activation
        x = self.dropout2(x)  # Apply dropout
        x = self.fc2(x)  # -> (batch, 10) Output logits

        return x
