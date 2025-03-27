# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/train.py
# Description: Script for training the MNIST digit classifier model.
# Created: Earlier Date (based on file listing)
# Updated: 2025-03-27 (Integrated evaluation, updated batch size, formatting)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets  # Removed 'transforms' as it's unused directly
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
import time
import datetime
import json

# Project-specific imports
try:
    from model.model import MNISTClassifier
    from model.utils.augmentation import get_train_transforms, get_test_transforms
    from model.utils.evaluation import (
        evaluate_model,
        plot_training_history,
        generate_evaluation_report  # Import the new function
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure this script is run correctly relative to the project root"
          " or the PYTHONPATH is set.")
    import sys
    sys.exit(1)

# Constants
RANDOM_SEED = 42
# Updated BATCH_SIZE based on MPS benchmark results
BATCH_SIZE = 512
NUM_EPOCHS = 15  # Default epochs, can be overridden by args
LEARNING_RATE = 0.001
SAVE_PATH = "model/saved_models/mnist_classifier.pt"
LOG_DIR = "model/logs"  # Log within the model directory
HISTORY_PLOT_PATH = 'training_history.png'  # Plot saved in root for easy access
FINAL_CM_PATH = 'final_confusion_matrix.png'
FINAL_REPORT_PATH = 'final_classification_report.txt'


def setup_logging(log_dir=LOG_DIR):
    """Set up logging to file and console with timestamps.

    Args:
        log_dir (str): Directory to save log files.

    Returns:
        str: Path to the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )

    logging.info(f"Logging setup complete. Log file: {log_file}")
    return log_file


def set_seed(seed):
    """Set random seeds for reproducibility across libraries.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    np.random.seed(seed)
    # random.seed(seed) # Consider if stdlib random is used elsewhere
    # Ensure deterministic algorithms are used where possible
    # Note: MPS backend might not have full deterministic support
    # torch.backends.cudnn.deterministic = True (if using CUDA)
    # torch.backends.cudnn.benchmark = False (if using CUDA)
    logging.info(f"Set random seed to {seed}")


def get_device():
    """Determine and return the optimal device for PyTorch operations.

    Prioritizes MPS (Apple Silicon GPU), then CUDA, then CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        # Consider logging thread count if relevant: torch.get_num_threads()
        logging.info("Using CPU")
    return device


def get_data_loaders(batch_size, num_workers=4, pin_memory=True, data_dir='./data'):
    """Create train and test data loaders for MNIST.

    Applies appropriate transformations (including augmentation for training).

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
            Recommended to set to 0 for MPS.
        pin_memory (bool): If True, copies Tensors into device/CUDA pinned
                           memory before returning them. Often False for MPS.
        data_dir (str): Directory where MNIST data is stored/downloaded.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Adjust num_workers and pin_memory for MPS compatibility
    if get_device().type == 'mps':
        num_workers = 0
        pin_memory = False
        logging.info("Adjusted DataLoader: num_workers=0, pin_memory=False "
                     "for MPS.")

    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    try:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
    except Exception as e:
        logging.error(f"Failed to load MNIST dataset from {data_dir}: {e}")
        raise

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    logging.info(f"Data loaders created with batch size {batch_size}.")
    return train_loader, test_loader


def train_model(num_epochs, batch_size, learning_rate, device,
                save_path, log_dir):
    """Trains the MNIST classifier model.

    Args:
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Training batch size.
        learning_rate (float): Optimizer learning rate.
        device (torch.device): Device to train on (CPU, CUDA, MPS).
        save_path (str): Path to save the best model checkpoint.
        log_dir (str): Directory to save training logs and history.

    Returns:
        tuple: (path_to_best_model, training_history)
    """
    start_time = time.time()

    # --- Setup ---
    train_loader, test_loader = get_data_loaders(batch_size)
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler reduces LR on plateauing validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5, verbose=True
    )
    logging.info(f"Model, Loss, Optimizer, Scheduler initialized on {device}.")

    # --- Training Loop ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = -1

    logging.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar postfix
            current_loss = loss.item()
            current_acc = correct_train / total_train if total_train > 0 else 0.0
            train_pbar.set_postfix({
                'Loss': f"{current_loss:.4f}",
                'Acc': f"{current_acc:.4f}"
            })

        # Calculate epoch statistics for training
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()  # Set model to evaluation mode
        epoch_val_loss, epoch_val_acc = evaluate_model(
            model, test_loader, criterion, device
        )

        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)

        # Log epoch results
        epoch_duration = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f} | "
            f"Duration: {epoch_duration:.2f}s"
        )

        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Save the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logging.info(
                f"🎉 New best model saved at Epoch {best_epoch} "
                f"with Val Loss: {best_val_loss:.4f}"
            )

    # --- Post-Training ---
    total_training_time = time.time() - start_time
    logging.info(f"🏁 Training finished in {total_training_time:.2f} seconds.")
    logging.info(f"🏆 Best model saved from Epoch {best_epoch} "
                 f"at {save_path}")

    # Plot training history
    try:
        plot_training_history(history, save_path=HISTORY_PLOT_PATH)
        logging.info(f"📈 Training history plot saved to {HISTORY_PLOT_PATH}")
    except Exception as e:
        logging.warning(f"Could not plot training history: {e}")

    # Save history dictionary
    history_file_name = f"training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    history_file_path = os.path.join(log_dir, history_file_name)
    try:
        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=4)
        logging.info(f"💾 Training history data saved to {history_file_path}")
    except Exception as e:
        logging.warning(f"Could not save training history JSON: {e}")

    return save_path, history


def final_evaluation(model_path, device, test_loader):
    """Performs final evaluation using the best saved model.

    Generates and saves the confusion matrix and classification report.

    Args:
        model_path (str): Path to the saved best model state_dict.
        device (torch.device): Device to run evaluation on.
        test_loader (DataLoader): DataLoader for the test dataset.
    """
    logging.info("\n" + "="*30 + "\n🔬 Performing Final Evaluation 🔬\n" + "="*30)
    model = MNISTClassifier().to(device)
    try:
        # Load the best model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info(f"Loaded best model from {model_path} for final eval.")
    except Exception as e:
        logging.error(f"Failed to load best model for final evaluation: {e}")
        return

    # Generate the report and confusion matrix
    try:
        generate_evaluation_report(
            model=model,
            data_loader=test_loader,
            device=device,
            save_cm_path=FINAL_CM_PATH,
            save_report_path=FINAL_REPORT_PATH
        )
        logging.info(f"📊 Final evaluation reports saved: "
                     f"{FINAL_CM_PATH}, {FINAL_REPORT_PATH}")
    except Exception as e:
        logging.error(f"Failed during final evaluation report generation: {e}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train MNIST Classifier Model")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help=f"Path to save best model (default: {SAVE_PATH})")
    parser.add_argument('--log_dir', type=str, default=LOG_DIR,
                        help=f"Directory for logs (default: {LOG_DIR})")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    # --- Main Execution ---
    log_file = setup_logging(args.log_dir)
    set_seed(args.seed)
    selected_device = get_device()

    # Start Training
    best_model_path, training_history = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=selected_device,
        save_path=args.save_path,
        log_dir=args.log_dir
    )

    # Perform Final Evaluation (using the best model saved during training)
    # Re-create test loader just for final eval if needed, or reuse
    _, final_test_loader = get_data_loaders(args.batch_size) # Use trained BS
    final_evaluation(best_model_path, selected_device, final_test_loader)

    logging.info("✅ Training script finished.")