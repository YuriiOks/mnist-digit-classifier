import requests
import base64
import io
import json
from PIL import Image

class ModelClient:
    """
    Client for interacting with the model service.
    """
    
    def __init__(self, base_url):
        """
        Initialize the model client.
        
        Args:
            base_url (str): Base URL of the model service
        """
        self.base_url = base_url
    
    def health_check(self):
        """
        Check if the model service is healthy.
        
        Returns:
            bool: True if the service is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def predict(self, image):
        """
        Get a prediction from the model.
        
        Args:
            image: PIL Image to predict
            
        Returns:
            tuple: (predicted_digit, confidence)
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Send to model service
        response = requests.post(
            f"{self.base_url}/predict",
            json={"image": img_str},
            timeout=5
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            return result["prediction"], result["confidence"]
        else:
            error = response.json().get("error", "Unknown error")
            raise Exception(f"Model service error: {error}") 