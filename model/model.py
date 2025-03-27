import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        # Add BatchNorm after conv1
        self.bn1 = nn.BatchNorm2d(32) # Match num_features from conv1 output
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        # Add BatchNorm after conv2
        self.bn2 = nn.BatchNorm2d(64) # Match num_features from conv2 output
        self.dropout1 = nn.Dropout2d(0.25) # Original Dropout uses Dropout2d
        self.dropout2 = nn.Dropout(0.5)    # Original Dropout uses Dropout (for FC layers)
        # Calculate the input size for fc1 carefully after pooling
        # If input is 28x28 -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> pool -> dropout -> pool
        # 28x28 -> 28x28 -> 28x28 -> 14x14 -> 14x14 -> 7x7
        # So the flattened size is 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Pool 1: 28x28 -> 14x14

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Pool 2: 14x14 -> 7x7

        # Apply Dropout *after* convolutions and pooling
        # Dropout2d is suitable here as it works on feature maps
        x = self.dropout1(x)

        # Flatten for FC layers
        # Now the shape should be Batch x 64 x 7 x 7
        x = torch.flatten(x, 1) # Shape: Batch x 3136

        # FC Block
        x = self.fc1(x) # Should work now: Input is 3136, fc1 expects 3136
        x = F.relu(x)
        # Use regular Dropout for FC layers
        x = self.dropout2(x)
        x = self.fc2(x)

        return x # Return raw logits