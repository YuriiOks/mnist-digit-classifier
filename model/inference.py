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
        # Set device - check for MPS (Apple Silicon) first
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

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

            # For MPS device, synchronize to ensure computation is complete
            if self.device.type == "mps":
                torch.mps.synchronize()

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
            confidences = [
                probabilities[i, cls].item() for i, cls in enumerate(predicted_classes)
            ]

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
            image = Image.open(image_path).convert("L")
            digit, confidence = predict_digit(image)
            print(f"Predicted Digit: {digit}")
            print(f"Confidence: {confidence:.2%}")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please provide an image path")
