from torchvision import transforms
import torch

def get_train_transforms():
    """
    Returns data transformations for training data with augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations for training
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

def get_test_transforms():
    """
    Returns data transformations for test data (no augmentation).
    
    Returns:
        torchvision.transforms.Compose: Composed transformations for testing
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

def get_enhanced_train_transforms():
    """
    Get enhanced data augmentation for better generalization.
    
    This is useful if the model struggles with real-world drawn digits.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.15, 0.15), 
            scale=(0.85, 1.15), 
            shear=10
        ),
        # ElasticTransform simulates handwriting variations
        transforms.ElasticTransform(alpha=1.0, sigma=5.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])

def get_canvas_simulation_transforms():
    """
    Get transforms that simulate browser canvas input.
    
    These transforms try to mimic the variations we might see in 
    digits drawn on a web canvas.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2), 
            scale=(0.8, 1.2), 
            shear=15
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ElasticTransform(alpha=2.0, sigma=8.0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
