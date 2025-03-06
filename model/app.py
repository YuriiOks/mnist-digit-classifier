import os
import base64
import io
import json
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

# Define the model architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load or initialize the model
model_path = os.environ.get('MODEL_PATH', 'saved_models/mnist_classifier.pt')

model = MNISTClassifier()
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Could not load model: {e}")
    print("Using untrained model (predictions will be random)")

model.eval()

# Image preprocessing
def preprocess_image(image_data):
    # Convert base64 to PIL Image
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Center the digit
        # Find bounding box of non-zero pixels
        bbox = Image.eval(image, lambda px: 255 if px < 200 else 0).getbbox()
        
        if bbox is not None:
            # Crop to bounding box
            image = image.crop(bbox)
        
        # Resize and pad to get a square image
        size = max(image.size)
        new_im = Image.new('L', (size, size), 255)
        new_im.paste(image, ((size - image.size[0]) // 2, (size - image.size[1]) // 2))
        
        # Resize to 28x28
        image = new_im.resize((28, 28))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

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
            
            # Preprocess image
            image_tensor = preprocess_image(data['image'])
            if image_tensor is None:
                return jsonify({"error": "Invalid image data"}), 400
            
        elif request.content_type.startswith('multipart/form-data'):
            # Handle form upload
            if 'image' not in request.files:
                return jsonify({"error": "No image file in request"}), 400
            
            file = request.files['image']
            image_tensor = preprocess_image(file.read())
            
        elif request.content_type.startswith('image/'):
            # Handle binary image data
            image_data = request.data
            image_tensor = preprocess_image(base64.b64encode(image_data).decode('utf-8'))
        
        else:
            return jsonify({"error": "Unsupported content type"}), 415
        
        if image_tensor is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Return prediction
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)
