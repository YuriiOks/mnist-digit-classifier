# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: scripts/visualize_augmentations.py
# Description: Script for visualizing the effect of different data augmentation
# Created: 2025-03-27
# Updated: 2025-03-27

import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_pil_image
from typing import Tuple, Any

# Dynamically add project root to sys.path to ensure imports work correctly
# when run directly or as a module from the project root.
try:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    if project_root not in sys.path:
        print(f"Adding project root to sys.path: {project_root}")
        sys.path.insert(0, project_root)
    # Now try the import
    from model.utils.augmentation import (
        get_train_transforms,
        get_test_transforms,
        get_enhanced_train_transforms,
        get_canvas_simulation_transforms,
    )

    print("Successfully imported augmentation functions.")
except ImportError as e:
    print(
        f"CRITICAL ERROR: Failed to import augmentation functions: {e}",
        file=sys.stderr,
    )
    print("Please ensure:", file=sys.stderr)
    print(f"  - Project Root: '{project_root}' is correct.", file=sys.stderr)
    print(
        "  - 'utils' directory exists at the project root.", file=sys.stderr
    )
    print(
        "  - 'utils' contains '__init__.py' and 'augmentation.py'.",
        file=sys.stderr,
    )
    print(
        "  - This script is run from the project root (e.g., using 'python -m scripts.visualize_augmentations').",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as e:
    print(
        f"An unexpected error occurred during initial setup: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

# Constants
# Assume MNIST data is downloaded/stored within the 'model/data' subdirectory
MNIST_DATA_PATH = os.path.join(project_root, "model", "data")
OUTPUT_DIR = os.path.join(project_root, "outputs", "figures")
MNIST_MEAN = (0.1307,)  # Standard MNIST mean
MNIST_STD = (0.3081,)  # Standard MNIST standard deviation

# Helper Functions


def get_transform_pipeline(transform_type: str) -> Tuple[Any, str]:
    """
    Selects and returns the appropriate augmentation pipeline based on the type.

    Args:
        transform_type: A string identifier for the desired transform
                        ('train', 'test', 'enhanced', 'canvas').

    Returns:
        A tuple containing the torchvision transform composition and a
        descriptive title string for the pipeline.

    Raises:
        ValueError: If an unknown transform_type is provided.
    """
    if transform_type == "train":
        return get_train_transforms(), "Standard Training Augmentation"
    elif transform_type == "enhanced":
        return get_enhanced_train_transforms(), "Enhanced Augmentation"
    elif transform_type == "canvas":
        return (
            get_canvas_simulation_transforms(),
            "Canvas Simulation Augmentation",
        )
    elif transform_type == "test":
        return get_test_transforms(), "Test/Validation (No Augmentation)"
    else:
        raise ValueError(
            f"Unknown transform_type specified: {transform_type}"
        )


def unnormalize_image(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverses the normalization applied to an image tensor using MNIST statistics.

    Args:
        img_tensor: A normalized image tensor (torch.Tensor) typically with
                    shape [C, H, W].

    Returns:
        A torch.Tensor containing the un-normalized image data, clamped to the
        valid [0, 1] range.
    """
    # Ensure mean and std tensors have the correct shape for broadcasting ([C, 1, 1])
    mean = torch.tensor(MNIST_MEAN).view(-1, 1, 1)
    std = torch.tensor(MNIST_STD).view(-1, 1, 1)
    # Apply the inverse transformation: Original = (Normalized * Std) + Mean
    img_unnorm = img_tensor * std + mean
    # Clamp values to ensure they are within the valid [0, 1] image range
    return torch.clamp(img_unnorm, 0, 1)


# Main Visualization Logic


def visualize_augmentations(transform_type: str, num_samples: int, seed: int):
    """
    Loads MNIST dataset, applies a specified augmentation pipeline to random samples,
    and saves a visualization comparing original and augmented images.

    Args:
        transform_type: Identifier for the augmentation pipeline to use
                        ('train', 'test', 'enhanced', 'canvas').
        num_samples: The number of random samples to visualize.
        seed: An integer seed for random number generation to ensure reproducibility.
    """
    print(f"\n--- Starting Augmentation Visualization ---")
    print(f"Type: '{transform_type}', Samples: {num_samples}, Seed: {seed}")

    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Get the selected transform pipeline and descriptive title
    try:
        transform_pipeline, title_suffix = get_transform_pipeline(
            transform_type
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load MNIST training data - get raw PIL images by not applying transforms here
    try:
        # Ensure parent directory exists for download if needed
        os.makedirs(os.path.dirname(MNIST_DATA_PATH), exist_ok=True)
        raw_train_dataset = MNIST(
            root=MNIST_DATA_PATH, train=True, download=True
        )
        print(f"Loaded MNIST data from/to: {MNIST_DATA_PATH}")
        if len(raw_train_dataset) == 0:
            print(
                "Warning: MNIST dataset appears empty after loading.",
                file=sys.stderr,
            )
            return  # Cannot proceed if dataset is empty
    except Exception as e:
        print(
            f"Error loading MNIST dataset from {MNIST_DATA_PATH}: {e}",
            file=sys.stderr,
        )
        print("Please check the path and permissions.", file=sys.stderr)
        return

    # Prepare for plotting
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
    output_filename = f"augmentation_visualization_{transform_type}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Adjust num_samples if requesting more than available
    actual_num_samples = min(num_samples, len(raw_train_dataset))
    if actual_num_samples < num_samples:
        print(
            f"Warning: Requested {num_samples} samples, but dataset only has {len(raw_train_dataset)}. Visualizing {actual_num_samples}.",
            file=sys.stderr,
        )
    if actual_num_samples == 0:
        print(
            "Error: No samples available in the dataset to visualize.",
            file=sys.stderr,
        )
        return

    # Select random indices
    indices = random.sample(range(len(raw_train_dataset)), actual_num_samples)

    # Create plot
    fig, axes = plt.subplots(
        actual_num_samples, 2, figsize=(6, 2.5 * actual_num_samples)
    )
    # Handle case where num_samples=1, making axes accessible by index
    if actual_num_samples == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle(f"Original vs. {title_suffix}", fontsize=14, y=1.0)

    print("Generating plot...")
    for i, idx in enumerate(indices):
        # Get raw image (PIL format) and its label
        raw_img_pil, label = raw_train_dataset[idx]

        # Apply the selected augmentation pipeline (PIL -> Tensor)
        augmented_img_tensor = transform_pipeline(raw_img_pil)

        # Un-normalize the augmented tensor for correct display (Tensor -> Tensor)
        augmented_img_tensor_unnorm = unnormalize_image(augmented_img_tensor)

        # Convert un-normalized tensor back to PIL Image for plotting
        augmented_img_pil = to_pil_image(augmented_img_tensor_unnorm)

        # Plot Original Image
        ax_orig = axes[i, 0]
        ax_orig.imshow(raw_img_pil, cmap="gray", interpolation="nearest")
        ax_orig.set_title(f"Original (Label: {label})")
        ax_orig.axis("off")

        # Plot Augmented Image
        ax_aug = axes[i, 1]
        ax_aug.imshow(augmented_img_pil, cmap="gray", interpolation="nearest")
        ax_aug.set_title(f"Augmented ({transform_type})")
        ax_aug.axis("off")

    # Adjust layout and save
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.96]
    )  # Adjust rect to prevent title overlap
    try:
        plt.savefig(output_path)
        print(f"✅ Visualization saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}", file=sys.stderr)

    plt.close(fig)  # Close the figure explicitly to free up memory
    print("--- Visualization complete ---")


# Script Execution Guard
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Visualize MNIST Data Augmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        default="train",
        choices=["train", "test", "enhanced", "canvas"],
        help="Type of transform pipeline to visualize.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of random sample images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection and augmentation reproducibility.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    visualize_augmentations(
        transform_type=args.transform_type,
        num_samples=args.num_samples,
        seed=args.seed,
    )
