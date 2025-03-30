# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/utils/preprocessing.py
# Description: Preprocessing utilities for MNIST images, like centering.
# Created: 2025-03-06
# Updated: 2025-03-30

import logging
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from typing import Optional

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- MNIST Constants ---
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
# ---------------------


def center_digit(image: Image.Image) -> Image.Image:
    """
    Centers a white digit on a black background within a new square canvas.

    Assumes input image is PIL grayscale with a WHITE digit on a
    BLACK background (e.g., after padding). Uses bounding box method.

    Args:
        image: Input PIL Image (grayscale, White digit/Black bg expected).

    Returns:
        PIL Image: Image with the digit centered in a new black square canvas.
                   Returns the original image if centering fails.
    """
    try:
        img_gray = image.convert("L")  # Ensure grayscale

        # Find bounding box of the WHITE digit pixels (non-black pixels).
        try:
            bbox = img_gray.getbbox()  # Finds box containing non-zero pixels
        except ValueError:  # Handle potential errors on edge cases
            bbox = None

        if bbox is None:
            logger.warning(
                "center_digit: No bounding box found. " "Returning original."
            )
            return image

        # Crop the original image using the bounding box
        cropped_digit = img_gray.crop(bbox)
        digit_width, digit_height = cropped_digit.size

        # Create a new square canvas with BLACK background
        bg_color = 0  # Black background for centered output
        # Add generous padding around the largest dimension
        padding = 30
        new_size = max(digit_width, digit_height) + 2 * padding
        centered_canvas = Image.new("L", (new_size, new_size), bg_color)

        # Calculate top-left corner for pasting onto the center
        paste_x = (new_size - digit_width) // 2
        paste_y = (new_size - digit_height) // 2

        # Paste the cropped (white) digit onto the center of the black canvas
        centered_canvas.paste(cropped_digit, (paste_x, paste_y))

        logger.debug(
            f"center_digit: Centered W-on-B digit in new {new_size}x"
            f"{new_size} black canvas."
        )
        return centered_canvas

    except Exception as e:
        logger.error(
            f"💥 Error in center_digit: {e}. Returning original.", exc_info=True
        )
        return image


def resize_and_normalize(image: Image.Image) -> Optional[torch.Tensor]:
    """
    Resizes a PIL image to 28x28 and applies MNIST normalization.

    Assumes input is a grayscale PIL Image (e.g., after centering).

    Args:
        image: Input PIL Image.

    Returns:
        Normalized torch.Tensor of shape [1, 1, 28, 28] or None on failure.
    """
    try:
        # Note: Input to this should be White-on-Black for MNIST convention
        transform_pipeline = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  # White(255)->1.0; Black(0)->0.0
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )
        # Apply transforms and add batch dimension
        tensor = transform_pipeline(image).unsqueeze(0)
        return tensor
    except Exception as e:
        logger.error(f"💥 Error in resize_and_normalize: {e}", exc_info=True)
        return None
