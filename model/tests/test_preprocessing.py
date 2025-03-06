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
