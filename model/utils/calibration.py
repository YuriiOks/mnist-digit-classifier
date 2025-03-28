# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/utils/calibration.py
# Description: Utilities for model calibration analysis and implementation.
# Created: 2025-03-28

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm # For progress if loading data here

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Reliability Diagram Functions ---

def get_probabilities_labels(model: torch.nn.Module,
                             data_loader: torch.utils.data.DataLoader,
                             device: torch.device) -> tuple:
    """
    Gets model predicted probabilities and true labels for a dataset.

    Args:
        model: The PyTorch model (already trained).
        data_loader: DataLoader for the validation or test set.
        device: Device to run inference on.

    Returns:
        Tuple: (all_confidences, all_predictions, all_true_labels)
               Where all_confidences are the max probability for each sample,
               all_predictions are the predicted class index, and
               all_true_labels are the ground truth labels.
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_true_labels = []
    logger.info("📊 Getting probabilities and labels for calibration...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Calibration Data"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            # Get the highest probability (confidence) and predicted class
            confidences, predictions = torch.max(probabilities, dim=1)

            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    logger.info("✅ Probabilities and labels collected.")
    return np.array(all_confidences), \
           np.array(all_predictions), \
           np.array(all_true_labels)


def plot_reliability_diagram(confidences: np.ndarray,
                             predictions: np.ndarray,
                             true_labels: np.ndarray,
                             num_bins: int = 10,
                             save_path: str = None) -> float:
    """
    Calculates and plots a reliability diagram and returns the ECE.

    Args:
        confidences: Numpy array of confidence scores (max probability)
                     for each prediction.
        predictions: Numpy array of predicted class indices.
        true_labels: Numpy array of true class indices.
        num_bins: Number of bins to divide the confidence range into.
        save_path: Path to save the plot image. If None, plot is shown.

    Returns:
        Expected Calibration Error (ECE) score.
    """
    if not (len(confidences) == len(predictions) == len(true_labels)):
        raise ValueError("Input arrays must have the same length.")
    if len(confidences) == 0:
        logger.warning("Cannot plot reliability diagram: No data provided.")
        return float('nan') # Not a number indicates failure

    # --- Calculate Calibration Metrics ---
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    # Populate bins
    for i in range(num_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        bin_counts[i] = np.sum(in_bin)

        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(predictions[in_bin] == true_labels[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])

    # Calculate Expected Calibration Error (ECE)
    total_samples = len(confidences)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total_samples
    logger.info(f"📉 Calculated ECE: {ece:.4f}")

    # --- Plotting ---
    logger.info("📊 Plotting Reliability Diagram...")
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Reliability Diagram", fontsize=14)

    # Reliability Curve (Accuracy vs. Confidence)
    ax1 = axes[0]
    # Plot points only for bins with samples
    valid_bins = bin_counts > 0
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax1.bar(bin_lowers[valid_bins], bin_accuracies[valid_bins],
            width=(1.0/num_bins)*0.9, # Slight spacing between bars
            align='edge', edgecolor='black', label='Model Accuracy')
            # Use bar chart for binned accuracy

    # Add gaps visualization (optional)
    gap_points_x = bin_lowers[valid_bins] + (0.5 / num_bins)
    gap_points_y = bin_confidences[valid_bins]
    for i, idx in enumerate(np.where(valid_bins)[0]):
        ax1.plot([gap_points_x[i], gap_points_x[i]],
                 [bin_accuracies[idx], bin_confidences[idx]],
                 color='red', linestyle='-', linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.5)
    ax1.set_title(f"ECE = {ece:.4f}")

    # Confidence Histogram
    ax2 = axes[1]
    ax2.hist(confidences, bins=bin_boundaries, density=False,
             color='skyblue', edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax_twin = ax2.twinx() # Create a second y-axis for percentage
    ax_twin.set_ylabel('Frequency (%)')
    ax_twin.set_ylim(0, 100 * ax2.get_ylim()[1] / total_samples)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save or Show Plot
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"✅ Reliability Diagram saved to: {save_path}")
        except Exception as e:
            logger.error(f"🔥 Failed to save reliability diagram: {e}")
    else:
        plt.show()

    plt.close(fig) # Close figure to free memory
    return ece


# --- Temperature Scaling Functions (To be added next) ---

# def find_optimal_temperature(model, data_loader, device): ...
# class ModelWithTemperature(nn.Module): ...