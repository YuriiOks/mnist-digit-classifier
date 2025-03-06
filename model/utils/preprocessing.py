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
