import torch
from torchvision import transforms

def get_train_transforms():
    """
    Returns data transformations for training data with augmentation
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

def get_test_transforms():
    """
    Returns data transformations for test data (no augmentation)
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
