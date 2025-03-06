#!/bin/bash

# Create the directory structure
mkdir -p model/utils model/tests model/saved_models

# Create requirements.txt
cat > model/requirements.txt << 'EOF'
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
pillow==10.0.0
flask==2.3.2
scikit-image==0.21.0
pytest==7.3.1
matplotlib==3.7.1
tqdm==4.65.0
gunicorn==20.1.0
EOF

# Create model.py
cat > model/model.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    CNN model for MNIST digit classification.
    
    Architecture:
    - 2 convolutional layers with batch normalization
    - Max pooling
    - Dropout for regularization
    - 2 fully connected layers
    """
    
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Fully connected layers
        # Input size: 64 * 7 * 7 = 3136 (after 2 pooling operations)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Third conv block
        x = self.pool(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Return logits (not softmax - we'll use CrossEntropyLoss)
        return x
    
    def predict(self, x):
        """
        Return prediction and confidence scores.
        
        Args:
            x: Input tensor of shape [B, 1, 28, 28]
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence, _ = torch.max(probabilities, dim=1)
            
        return predicted_class, confidence
EOF

# Create train.py
cat > model/train.py << 'EOF'
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from model import MNISTClassifier
from utils.augmentation import get_train_transforms, get_test_transforms
from utils.evaluation import evaluate_model, plot_training_history

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
SAVE_PATH = "saved_models/mnist_classifier.pt"

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_data_loaders(batch_size):
    """
    Create train and test data loaders with appropriate transforms.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get transforms for training (with augmentation) and testing
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Load training dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_model():
    """Train the MNIST classifier model."""
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Initialize model
    model = MNISTClassifier().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for images, labels in train_pbar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate training statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        # Calculate epoch training statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST classifier")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()
    
    # Update parameters if provided
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    train_model()
EOF

# Create inference.py
cat > model/inference.py << 'EOF'
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from model import MNISTClassifier
from utils.preprocessing import preprocess_image

MODEL_PATH = "saved_models/mnist_classifier.pt"

class MNISTPredictor:
    """
    Class for loading the trained model and making predictions.
    
    This class handles:
    1. Loading the trained model
    2. Preprocessing input images
    3. Making predictions with confidence scores
    """
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model
        self.model = MNISTClassifier().to(self.device)
        
        # Load trained parameters
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, image):
        """
        Make a prediction from an image.
        
        Args:
            image: PIL Image or numpy array or base64 encoded image
            
        Returns:
            tuple: (predicted_digit, confidence)
        """
        # Preprocess the image
        tensor = preprocess_image(image)
        tensor = tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence
    
    def predict_batch(self, images):
        """
        Make predictions for a batch of images.
        
        Args:
            images: Batch of images (list of PIL Images or numpy arrays)
            
        Returns:
            tuple: (predicted_digits, confidences)
        """
        # Preprocess all images
        tensors = [preprocess_image(img) for img in images]
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).tolist()
            confidences = [probabilities[i, cls].item() for i, cls in enumerate(predicted_classes)]
        
        return predicted_classes, confidences

# Helper function for quick prediction
def predict_digit(image):
    """
    Quick helper function to predict a digit from an image.
    
    Args:
        image: PIL Image or numpy array or base64 encoded image
        
    Returns:
        tuple: (predicted_digit, confidence)
    """
    predictor = MNISTPredictor()
    return predictor.predict(image)

if __name__ == "__main__":
    import sys
    from PIL import Image
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            image = Image.open(image_path).convert('L')
            digit, confidence = predict_digit(image)
            print(f"Predicted Digit: {digit}")
            print(f"Confidence: {confidence:.2%}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please provide an image path")
EOF

# Create app.py
cat > model/app.py << 'EOF'
import os
import base64
import io
import json
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from inference import MNISTPredictor

app = Flask(__name__)

# Initialize the predictor
predictor = MNISTPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running."""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for digit prediction.
    
    Accepts:
    - JSON with base64 encoded image
    - Binary image data
    - Multipart form with image file
    
    Returns:
    - JSON with prediction and confidence
    """
    try:
        image = None
        
        # Handle different input methods
        if request.content_type == 'application/json':
            # Extract base64 encoded image from JSON
            data = request.get_json()
            if 'image' not in data:
                return jsonify({"error": "No image provided in JSON"}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data)).convert('L')
            
        elif request.content_type.startswith('multipart/form-data'):
            # Handle form upload
            if 'image' not in request.files:
                return jsonify({"error": "No image file in request"}), 400
            
            file = request.files['image']
            image = Image.open(file.stream).convert('L')
            
        elif request.content_type.startswith('image/'):
            # Handle binary image data
            image_data = request.data
            image = Image.open(io.BytesIO(image_data)).convert('L')
        
        else:
            return jsonify({"error": "Unsupported content type"}), 415
        
        if image is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Make prediction
        digit, confidence = predictor.predict(image)
        
        # Return prediction
        return jsonify({
            "prediction": int(digit),
            "confidence": float(confidence)
        })
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)
EOF

# Create utils/__init__.py
cat > model/utils/__init__.py << 'EOF'
# This file is intentionally left empty to make utils a Python package
EOF

# Create utils/preprocessing.py
cat > model/utils/preprocessing.py << 'EOF'
import torch
import numpy as np
from PIL import Image, ImageOps
import base64
import io
from skimage import transform, filters
import torchvision.transforms as transforms

def preprocess_image(input_image):
    """
    Preprocess an image for MNIST model inference.
    
    This function handles various input types and performs preprocessing to match
    the MNIST dataset distribution:
    1. Convert to grayscale
    2. Center the digit
    3. Resize to 28x28
    4. Normalize pixel values
    
    Args:
        input_image: PIL Image, numpy array, or base64 encoded image
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape [1, 1, 28, 28]
    """
    # Handle different input types
    if isinstance(input_image, str) and input_image.startswith('data:image'):
        # Handle data URL
        header, encoded = input_image.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert('L')
    elif isinstance(input_image, str):
        # Assume base64 encoded
        try:
            image_data = base64.b64decode(input_image)
            image = Image.open(io.BytesIO(image_data)).convert('L')
        except Exception:
            # If not base64, try as file path
            image = Image.open(input_image).convert('L')
    elif isinstance(input_image, bytes):
        # Raw image bytes
        image = Image.open(io.BytesIO(input_image)).convert('L')
    elif isinstance(input_image, np.ndarray):
        # Numpy array
        if input_image.ndim == 3 and input_image.shape[2] > 1:
            # Convert RGB to grayscale
            image = Image.fromarray(input_image).convert('L')
        else:
            image = Image.fromarray(input_image.astype(np.uint8)).convert('L')
    elif isinstance(input_image, Image.Image):
        # PIL Image
        image = input_image.convert('L')
    else:
        raise ValueError("Unsupported input type. Expected PIL Image, numpy array, or base64 string")
    
    # Process the image to match MNIST distribution
    image = center_digit(image)
    image = resize_and_normalize(image)
    
    return image

def center_digit(image):
    """
    Center the digit in the image using center of mass.
    
    Args:
        image: PIL Image in grayscale mode
        
    Returns:
        PIL Image: Centered image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Invert if white digit on black background
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Apply threshold to separate digit from background
    thresh = filters.threshold_otsu(img_array)
    binary = img_array > thresh
    
    # If no foreground pixels are found, return the original image
    if not np.any(binary):
        return image
    
    # Calculate center of mass
    rows, cols = np.where(binary)
    if len(rows) == 0 or len(cols) == 0:
        return image
    
    cy, cx = np.mean(rows), np.mean(cols)
    
    # Calculate center of image
    height, width = img_array.shape
    center_y, center_x = height // 2, width // 2
    
    # Calculate translation needed
    dy, dx = center_y - cy, center_x - cx
    
    # Apply translation using skimage
    translation = transform.AffineTransform(translation=(dx, dy))
    translated = transform.warp(img_array, translation, mode='constant', preserve_range=True).astype(np.uint8)
    
    return Image.fromarray(translated)

def resize_and_normalize(image):
    """
    Resize the image to 28x28 and normalize for the model.
    
    Args:
        image: PIL Image
        
    Returns:
        torch.Tensor: Tensor of shape [1, 1, 28, 28]
    """
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Apply transforms
    tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    return tensor

def normalize_stroke_width(image, target_width=2):
    """
    Normalize the stroke width of a digit.
    
    Args:
        image: PIL Image
        target_width: Target stroke width in pixels
        
    Returns:
        PIL Image: Image with normalized stroke width
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure image is binary
    thresh = filters.threshold_otsu(img_array)
    binary = (img_array > thresh).astype(np.uint8) * 255
    
    # Determine current stroke width through distance transform
    dist_transform = transform.distance_transform_edt(binary)
    current_width = np.max(dist_transform) * 2  # Diameter
    
    # Skip if current width is close to target
    if abs(current_width - target_width) < 0.5:
        return image
    
    # Apply morphological operations to adjust width
    if current_width > target_width:
        # Erode to reduce width
        from skimage.morphology import erosion, disk
        iterations = int(round((current_width - target_width) / 2))
        selem = disk(1)
        for _ in range(iterations):
            binary = erosion(binary, selem)
    else:
        # Dilate to increase width
        from skimage.morphology import dilation, disk
        iterations = int(round((target_width - current_width) / 2))
        selem = disk(1)
        for _ in range(iterations):
            binary = dilation(binary, selem)
    
    return Image.fromarray(binary)
EOF

# Create utils/augmentation.py
cat > model/utils/augmentation.py << 'EOF'
from torchvision import transforms
import torch

def get_train_transforms():
    """
    Get data augmentation transforms for training.
    
    The transforms include:
    - Random rotation
    - Random affine transformation (slight distortion)
    - Random erasing (simulates noise)
    - Normalization
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1), 
            shear=5
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

def get_test_transforms():
    """
    Get transforms for testing/validation/inference.
    
    The transforms include only:
    - Conversion to tensor
    - Normalization
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms
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
EOF

# Create utils/evaluation.py
cat > model/utils/evaluation.py << 'EOF'
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = val_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def get_confusion_matrix(model, data_loader, device):
    """
    Generate confusion matrix for the model.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    model.eval()
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Generating Confusion Matrix'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)
    
    return cm

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_model_predictions(model, data_loader, device, num_images=25):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with samples
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(images.size(0)):
                images_so_far += 1
                ax = plt.subplot(5, 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'Pred: {preds[j]}, True: {labels[j]}')
                
                # Reverse normalization for display
                mean = torch.tensor([0.1307]).view(1, 1, 1)
                std = torch.tensor([0.3081]).view(1, 1, 1)
                img = images[j].cpu() * std + mean
                
                ax.imshow(img.squeeze().numpy(), cmap='gray')
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('model_predictions.png')
                    plt.show()
                    return
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()
EOF

# Create tests/__init__.py
cat > model/tests/__init__.py << 'EOF'
# This file is intentionally left empty to make tests a Python package
EOF

# Create tests/test_model.py
cat > model/tests/test_model.py << 'EOF'
import pytest
import torch
from model import MNISTClassifier

def test_model_architecture():
    """Test that the model architecture is as expected."""
    model = MNISTClassifier()
    
    # Check that we have the expected layers
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'conv2')
    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc2')
    
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
EOF

# Create tests/test_preprocessing.py
cat > model/tests/test_preprocessing.py << 'EOF'
import pytest
import torch
import numpy as np
from PIL import Image
import base64
import io
from utils.preprocessing import preprocess_image, center_digit, resize_and_normalize

def test_preprocess_image_with_pil():
    """Test preprocessing with PIL image input."""
    # Create a dummy image
    image = Image.new('L', (100, 100), 255)
    
    # Draw a simple digit-like shape
    for i in range(40, 60):
        for j in range(30, 70):
            image.putpixel((i, j), 0)
    
    # Preprocess the image
    tensor = preprocess_image(image)
    
    # Check the tensor shape
    assert tensor.shape == (1, 1, 28, 28)
    
    # Check that tensor values are normalized
    assert torch.min(tensor) < 0
    assert torch.max(tensor) > 0

def test_preprocess_image_with_numpy():
    """Test preprocessing with numpy array input."""
    # Create a dummy image as numpy array
    image = np.ones((100, 100), dtype=np.uint8) * 255
    
    # Draw a simple digit-like shape
    image[40:60, 30:70] = 0
    
    # Preprocess the image
    tensor = preprocess_image(image)
    
    # Check the tensor shape
    assert tensor.shape == (1, 1, 28, 28)

def test_preprocess_image_with_base64():
    """Test preprocessing with base64 encoded image."""
    # Create a dummy image
    image = Image.new('L', (100, 100), 255)
    
    # Draw a simple digit-like shape
    for i in range(40, 60):
        for j in range(30, 70):
            image.putpixel((i, j), 0)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Preprocess the image
    tensor = preprocess_image(base64_string)
    
    # Check the tensor shape
    assert tensor.shape == (1, 1, 28, 28)

def test_center_digit():
    """Test centering of a digit."""
    # Create an off-center digit
    image = Image.new('L', (100, 100), 255)
    
    # Draw digit in top-left quadrant
    for i in range(10, 30):
        for j in range(10, 30):
            image.putpixel((i, j), 0)
    
    # Center the digit
    centered = center_digit(image)
    
    # Convert to numpy for testing
    img_array = np.array(centered)
    
    # Find foreground pixels
    rows, cols = np.where(img_array < 127)
    
    if len(rows) > 0 and len(cols) > 0:
        # Calculate center of mass
        cy, cx = np.mean(rows), np.mean(cols)
        
        # Check that center of mass is close to image center
        height, width = img_array.shape
        assert abs(cy - height/2) < 10
        assert abs(cx - width/2) < 10

def test_resize_and_normalize():
    """Test resizing and normalizing an image."""
    # Create a dummy image
    image = Image.new('L', (100, 100), 255)
    
    # Draw a simple digit
    for i in range(40, 60):
        for j in range(30, 70):
            image.putpixel((i, j), 0)
    
    # Resize and normalize
    tensor = resize_and_normalize(image)
    
    # Check shape
    assert tensor.shape == (1, 1, 28, 28)
    
    # Check normalization - values should be centered around MNIST mean/std
    assert torch.min(tensor) < 0
    assert torch.max(tensor) > 0

if __name__ == "__main__":
    test_preprocess_image_with_pil()
    test_preprocess_image_with_numpy()
    test_preprocess_image_with_base64()
    test_center_digit()
    test_resize_and_normalize()
    print("All tests passed!")
EOF

# Create tests/test_inference.py
cat > model/tests/test_inference.py << 'EOF'
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
    reason="Trained model not found"
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
    image = Image.new('L', (28, 28), 255)
    
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
        image = Image.new('L', (28, 28), 255)
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
EOF

# Create Dockerfile
cat > model/Dockerfile << 'EOF'
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY . .

# Create directory for saved models if it doesn't exist
RUN mkdir -p saved_models

# Volume for persistent model storage
VOLUME /app/saved_models

# Expose the port for API
EXPOSE 5000

# Start the API server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
EOF

# Create a gitkeep file for saved_models directory
mkdir -p model/saved_models
touch model/saved_models/.gitkeep

echo "Model folder setup complete!"