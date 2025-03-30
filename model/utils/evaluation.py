# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/utils/evaluation.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-06
# Updated: 2025-03-30

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import time
import logging


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # No gradients needed for evaluation
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy


def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Training history plot saved to {save_path}")
    except Exception as e:
        try:
            import logging

            logging.warning(f"Failed to save training history plot: {e}")
        except ImportError:
            print(f"Warning: Failed to save training history plot: {e}")


def get_predictions_and_labels(model, data_loader, device):
    """
    Get model predictions and true labels for a dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        tuple: (all_predicted_labels, all_true_labels)
    """
    model.eval()
    all_predicted = []
    all_labels = []
    print("\n🔍 Getting predictions for evaluation...")  # Add user feedback
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating Batches"):  # Keep tqdm
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("✅ Predictions collected.")
    return all_predicted, all_labels


# Add a new function to generate and display reports
def generate_evaluation_report(
    model,
    data_loader,
    device,
    class_names=None,
    save_cm_path="confusion_matrix.png",
    save_report_path="classification_report.txt",
):
    """
    Generates and saves confusion matrix and classification report.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names for plotting
        save_cm_path: Path to save the confusion matrix plot
        save_report_path: Path to save the classification report text
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]  # Default MNIST classes

    # 1. Get predictions and labels
    all_preds, all_labels = get_predictions_and_labels(model, data_loader, device)

    # 2. Compute Confusion Matrix
    print("📊 Computing Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix computed.")

    # 3. Plot Confusion Matrix
    print(f"🎨 Plotting Confusion Matrix (saving to {save_cm_path})...")
    plot_confusion_matrix(
        cm, class_names, save_path=save_cm_path
    )  # Add save_path argument to plot function if needed
    print("Confusion Matrix plot saved.")

    # 4. Compute and Print Classification Report
    print("\n📋 Computing Classification Report...")
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("--- Classification Report ---")
    print(report)
    print("-----------------------------")

    # 5. Save Classification Report to file
    try:
        with open(save_report_path, "w") as f:
            f.write("--- Classification Report ---\n")
            f.write(report)
        print(f"Classification Report saved to {save_report_path}")
    except Exception as e:
        print(f"⚠️ Error saving classification report: {e}")

    # Optional: Return the report and CM if needed elsewhere
    return cm, report


# You might need to slightly modify plot_confusion_matrix to accept a save_path
def plot_confusion_matrix(
    cm, class_names=None, save_path="confusion_matrix.png"
):  # Added save_path
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    try:
        plt.savefig(save_path)  # Use save_path
        # plt.show() # Optionally comment out or remove plt.show() for non-interactive runs
        plt.close()  # Close the figure
    except Exception as e:
        print(f"⚠️ Error saving confusion matrix plot: {e}")


def visualize_model_predictions(model, data_loader, device, num_images=25):
    """
    Visualize model predictions on sample images.

    Args:
        model: PyTorch model
        data_loader: DataLoader with samples
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for j in range(images.size(0)):
                images_so_far += 1
                ax = plt.subplot(5, 5, images_so_far)
                ax.axis("off")
                ax.set_title(f"Pred: {preds[j]}, True: {labels[j]}")

                # Reverse normalization for display
                mean = torch.tensor([0.1307]).view(1, 1, 1)
                std = torch.tensor([0.3081]).view(1, 1, 1)
                img = images[j].cpu() * std + mean

                ax.imshow(img.squeeze().numpy(), cmap="gray")

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig("model_predictions.png")
                    plt.show()
                    return

    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()
