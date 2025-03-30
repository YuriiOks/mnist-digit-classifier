# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/utils/calibration.py
# Description: Utilities for model calibration analysis and temperature scaling.
# Created: 2025-03-28
# Updated: 2025-03-30

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_probabilities_labels(
    model: nn.Module, data_loader: DataLoader, device: torch.device
) -> tuple | None:
    """Gets model outputs (probabilities, predictions) and true labels.

    Runs inference on the provided data_loader and calculates softmax
    probabilities, predicted classes (argmax), and collects true labels.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for validation or test dataset.
        device: Device for inference (e.g., 'cpu', 'mps', 'cuda').

    Returns:
        Tuple (confidences, predictions, true_labels) as NumPy arrays,
        or None if dataloader is empty or an error occurs.
    """
    if not data_loader or len(data_loader.dataset) == 0:
        logger.warning("get_probabilities_labels: DataLoader is empty.")
        return None

    model.eval()
    all_confidences, all_predictions, all_true_labels = [], [], []
    logger.info("📊 Getting probabilities and labels for calibration...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Calibration Data"):
            try:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)

                all_confidences.append(confidences.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_true_labels.append(labels.cpu().numpy())
            except Exception as e:
                logger.error(
                    f"Error processing batch in " f"get_probabilities_labels: {e}"
                )
                return None  # Abort on error

    try:
        # Concatenate lists of arrays into single arrays
        confidences_np = np.concatenate(all_confidences)
        predictions_np = np.concatenate(all_predictions)
        true_labels_np = np.concatenate(all_true_labels)
        logger.info("✅ Probabilities and labels collected.")
        return confidences_np, predictions_np, true_labels_np
    except ValueError as e_cat:
        # Handle potential errors if lists are empty or shapes mismatch
        logger.error(f"Error concatenating results: {e_cat}")
        return None


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    num_bins: int = 15,
    save_path: str = None,
) -> float:
    """Calculates ECE and plots a reliability diagram.

    Args:
        confidences: NumPy array of confidence scores (max probability).
        predictions: NumPy array of predicted class indices.
        true_labels: NumPy array of true class indices.
        num_bins: Number of bins for confidence ranges.
        save_path: Path to save the plot image. If None, plot is shown.

    Returns:
        Expected Calibration Error (ECE) score, or float('nan') on failure.
    """
    if not (len(confidences) == len(predictions) == len(true_labels)):
        logger.error("Input arrays must have the same length for reliability.")
        return float("nan")
    if len(confidences) == 0:
        logger.warning("Cannot plot reliability diagram: No data.")
        return float("nan")

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=int)

    for i in range(num_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(predictions[in_bin] == true_labels[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])

    # --- Calculate ECE ---
    total_samples = len(confidences)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total_samples
    logger.info(f"📉 Calculated ECE: {ece:.4f}")

    # --- Plotting ---
    logger.info("📊 Plotting Reliability Diagram...")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle("Reliability Diagram", fontsize=14)
    ax1, ax2 = axes

    # Top Plot: Reliability Curve
    valid_bins = bin_counts > 0
    bar_centers = bin_lowers + (0.5 / num_bins)  # Center bars in bins
    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration",
    )
    ax1.bar(
        bin_lowers[valid_bins],
        bin_accuracies[valid_bins],
        width=(1.0 / num_bins) * 0.9,
        align="edge",
        edgecolor="black",
        label="Model Accuracy",
    )
    # Plot gaps (difference between confidence and accuracy)
    gap_points_y = bin_confidences[valid_bins]
    for i, idx in enumerate(np.where(valid_bins)[0]):
        ax1.plot(
            [bar_centers[idx], bar_centers[idx]],
            [bin_accuracies[idx], gap_points_y[i]],
            color="red",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
        )

    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.5)
    ax1.set_title(f"ECE = {ece:.4f}")

    # Bottom Plot: Confidence Histogram
    ax2.hist(
        confidences,
        bins=bin_boundaries,
        density=False,
        color="skyblue",
        edgecolor="black",
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax_twin = ax2.twinx()
    ax_twin.set_ylabel("Frequency (%)")
    ax_twin.set_ylim(
        0,
        100 * ax2.get_ylim()[1] / total_samples if total_samples > 0 else 100,
    )
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"✅ Reliability Diagram saved to: {save_path}")
        except Exception as e:
            logger.error(f"🔥 Failed to save reliability diagram: {e}")
    else:
        plt.show()  # Show plot if no save path provided

    plt.close(fig)
    return ece


class _ECELoss(nn.Module):
    """Helper nn.Module to calculate Expected Calibration Error."""

    def __init__(self, n_bins=15):
        """Initialize the _ECELoss instance."""
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.n_bins = n_bins

    def forward(self, logits, labels):
        """Computes ECE from logits and labels."""
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        for i in range(self.n_bins):
            in_bin = confidences.gt(self.bin_lowers[i]) & confidences.le(
                self.bin_uppers[i]
            )
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ModelWithTemperature(nn.Module):
    """A wrapper nn.Module for temperature scaling."""

    def __init__(self, model: nn.Module):
        """Initializes the wrapper with the base model."""
        super(ModelWithTemperature, self).__init__()
        if not model:
            raise ValueError("ModelWithTemperature requires a valid model.")
        self.model = model
        # Temperature is a parameter optimized on the validation set
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # Start > 1.0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies the base model and temperature scaling."""
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by the temperature parameter."""
        # Ensure temperature is positive during scaling
        temperature = self.temperature.clamp(min=1e-6)
        return logits / temperature

    def set_temperature(self, val_loader: DataLoader, device: torch.device) -> float:
        """Finds optimal temperature by minimizing NLL on validation set.

        Args:
            val_loader: DataLoader for the validation set.
            device: Device for calculations.

        Returns:
            The optimized temperature value.
        """
        if not val_loader or len(val_loader.dataset) == 0:
            logger.error("Validation loader is empty, cannot set temperature.")
            return self.temperature.item()  # Return current T

        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = _ECELoss().to(device)

        # Collect all logits and labels from validation set first
        logger.info("🌡️ Collecting logits/labels for temperature tuning...")
        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Temp Tuning Data"):
                images = images.to(device)
                logits = self.model(images)  # Use base model logits
                logits_list.append(logits)
                labels_list.append(labels)
        try:
            all_logits = torch.cat(logits_list).to(device)
            all_labels = torch.cat(labels_list).to(device)
        except (RuntimeError, ValueError) as e_cat:
            logger.error(
                f"Failed to gather logits/labels for temp scaling:"
                f" {e_cat}. Aborting tuning."
            )
            return self.temperature.item()

        logger.info("✅ Logits/labels collected.")

        # Calculate ECE before scaling
        try:
            ece_before = ece_criterion(all_logits, all_labels).item()
            logger.info(f"📊 ECE Before Temp Scaling: {ece_before:.4f}")
        except Exception as e_ece_before:
            logger.warning(f"Could not calculate initial ECE: {e_ece_before}")
            ece_before = float("inf")

        # Optimize temperature using LBFGS on NLL
        # LBFGS is efficient for optimizing single parameters like temperature
        optimizer = optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )

        def eval_nll_closure():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(all_logits)
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            logger.debug(
                f"  Temp: {self.temperature.item():.4f}, " f"NLL: {loss.item():.4f}"
            )
            return loss

        logger.info("🌡️ Optimizing temperature (minimizing NLL)...")
        optimizer.step(eval_nll_closure)
        logger.info("✅ Temperature optimization finished.")

        # Final temperature and ECE after scaling
        optimal_temp = self.temperature.item()
        try:
            with torch.no_grad():
                scaled_logits_final = self.temperature_scale(all_logits)
                ece_after = ece_criterion(scaled_logits_final, all_labels).item()
            logger.info(
                f"📊 ECE After Temp Scaling : {ece_after:.4f} "
                f"(Optimal T={optimal_temp:.4f})"
            )
            if ece_after >= ece_before - 1e-4:  # Check if ECE meaningfully improved
                logger.warning(
                    "⚠️ Temperature scaling did not significantly "
                    "improve ECE. Final T may not be optimal."
                )
        except Exception as e_ece_after:
            logger.warning(f"Could not calculate final ECE: {e_ece_after}")

        # Detach temperature after optimization
        self.temperature = nn.Parameter(torch.tensor([optimal_temp]))
        return optimal_temp