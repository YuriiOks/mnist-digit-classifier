# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/utils/augmentation.py
# Description: Data augmentation pipelines for MNIST.
# Created: Earlier Date
# Updated: 2025-03-27 (Implemented prioritized augmentation strategy)

from torchvision import transforms
import torch  # Needed for potential custom noise transforms later

# MNIST Mean and Standard Deviation (calculated from training set)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_train_transforms():
    """
    Returns data transformations for the training dataset.

    Applies a combination of realistic augmentations probabilistically
    to simulate variations in hand-drawn digits from canvas/uploads.
    Prioritizes geometric, elastic, perspective, blur, and intensity variations.

    Returns:
        torchvision.transforms.Compose: Composed transformations for training.
    """
    # Start with an empty list to build the pipeline
    transform_list = []

    # --- Augmentations Applied to PIL Image (Before ToTensor) ---

    # 1. Intensity / Contrast Variations (High Impact)
    # Why: Simulates different lighting in uploads or non-pure black/white
    #      from canvas anti-aliasing or different drawing tools.
    transform_list.append(
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4)
            # Note: Hue/Saturation are irrelevant for grayscale MNIST
        ], p=0.5) # Apply intensity changes 50% of the time
    )

    # --- Geometric Augmentations (High Impact) ---

    # 2. Affine Transformations (Rotation, Translation, Scale, Shear)
    # Why: Fundamental variations in how users draw/position digits.
    transform_list.append(
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=20,         # +/- 20 degrees rotation
                translate=(0.15, 0.15), # Shift up to 15% horizontally/vertically
                scale=(0.8, 1.2),   # Zoom between 80% and 120%
                shear=15            # Shear up to +/- 15 degrees
                # fill=0 -> default, fill with black if needed
            )
        ], p=0.8) # Apply affine transforms 80% of the time
    )

    # 3. Perspective Distortion (High Impact, esp. for uploads)
    # Why: Simulates viewing/capturing the digit from a slight angle.
    transform_list.append(
        transforms.RandomApply([
            transforms.RandomPerspective(
                distortion_scale=0.2, # Moderate distortion amount
                p=1.0 # Perspective distortion effect itself within RandomApply
            )
        ], p=0.3) # Apply perspective distortion 30% of the time
    )

    # --- Deformation / Quality Augmentations (High Impact) ---

    # 4. Elastic Deformation (Crucial for hand-drawing simulation)
    # Why: Mimics wobbly lines, local distortions from hand jitter/movement.
    transform_list.append(
        transforms.RandomApply([
            transforms.ElasticTransform(
                alpha=35.0, # Increase displacement magnitude slightly
                sigma=4.5   # Keep smoothness moderate for 28x28
            )
        ], p=0.5) # Apply elastic deformation 50% of the time
    )

    # 5. Gaussian Blur (Simulates out-of-focus, thick strokes, anti-alias)
    # Why: Addresses focus issues in uploads, varying stroke appearance.
    transform_list.append(
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=3,     # Small kernel for subtle blur
                sigma=(0.1, 1.2)   # Random sigma for varying blur intensity
            )
        ], p=0.4) # Apply blur 40% of the time
    )

    # --- Potential Medium Impact Augmentations (Optional - Add if needed later) ---

    # 6. Random Erasing (Simulates broken/disconnected strokes)
    # Why: Mimics quick drawing, pen lifts, minor obstructions.
    # transform_list.append(
    #     transforms.RandomApply([
    #         transforms.RandomErasing(
    #             p=1.0,           # Probability within RandomApply
    #             scale=(0.02, 0.1), # Erase 2% to 10% of the area
    #             ratio=(0.3, 3.3),  # Aspect ratio of the erased area
    #             value=0          # Fill with black
    #         )
    #     ], p=0.25) # Apply erasing 25% of the time
    # )

    # 7. Additive Gaussian Noise (Simulates sensor noise, grain)
    # Why: Models noise from low-quality uploads or image sensors.
    # Requires a custom Lambda transform:
    # transform_list.append(
    #     transforms.RandomApply([
    #         transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05) # Small noise std dev
    #     ], p=0.2) # Apply noise 20% of the time
    #     # Note: Noise should ideally be added *after* ToTensor if using torch.randn
    # )


    # --- Mandatory Final Steps (Convert to Tensor & Normalize) ---
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))

    # Compose all selected transformations into a single pipeline
    return transforms.Compose(transform_list)


def get_test_transforms():
    """
    Returns data transformations for the test/validation dataset.

    Only includes ToTensor and normalization (no augmentation).

    Returns:
        torchvision.transforms.Compose: Composed transformations for testing.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

# --- Deprecated / Experimental Functions ---
# Consider removing these if get_train_transforms is now the primary one.
# def get_canvas_simulation_transforms(): ...
# def get_enhanced_train_transforms(): ...