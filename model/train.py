# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/train.py
# Description: Script for training the MNIST digit classifier model.
# Created: Earlier Date
# Updated: 2025-03-28 (Added reliability diagram plotting)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
import time
import datetime
import json
import sys  # Keep sys for exit on import error

# Project-specific imports with error handling
try:
    from model.model import MNISTClassifier
    from model.utils.augmentation import get_train_transforms, get_test_transforms
    from model.utils.evaluation import (
        evaluate_model,
        plot_training_history,
        generate_evaluation_report
    )
    logger = logging.getLogger(__name__) # Use logger after imports
    logger.info("✅ Successfully imported project modules.")
    from model.utils.calibration import (
        get_probabilities_labels,
        plot_reliability_diagram
    )
except ImportError as e:
    # Use print before logging might be configured
    print(f"🔥 CRITICAL ERROR importing project modules: {e}")
    print("Ensure script is run relative to project root "
          "(e.g., python -m model.train) or PYTHONPATH is set.")
    sys.exit(1)

# --- Constants ---
RANDOM_SEED = 42
BATCH_SIZE = 512       # Optimal for MPS based on benchmarks
NUM_EPOCHS = 15        # Default epochs (override via args)
LEARNING_RATE = 0.001
SAVE_PATH = "model/saved_models/mnist_classifier.pt" # Model save location
LOG_DIR = "model/logs" # Training logs directory
# Output paths (relative to project root)
OUTPUT_FIG_DIR = "outputs/figures"
OUTPUT_DEBUG_DIR = "outputs/debug_images"
HISTORY_PLOT_FILENAME = "training_history.png"
FINAL_CM_FILENAME = "final_confusion_matrix.png"
FINAL_REPORT_FILENAME = "final_classification_report.txt"
DEBUG_TEST_NORM_FILENAME = "dbg_test_input_normalized.png"
DEBUG_TEST_UNNORM_FILENAME = "dbg_test_input_unnormalized.png"
RELIABILITY_PLOT_BEFORE_FILENAME = "reliability_diagram_before_calib.png"
RELIABILITY_PLOT_AFTER_FILENAME = "reliability_diagram_after_calib.png" 
# MNIST stats needed for un-normalization in debug save
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
# -----------------

def setup_logging(log_dir=LOG_DIR) -> str:
    """Sets up logging to file and console with timestamps.

    Args:
        log_dir: Directory to save log files.

    Returns:
        Path to the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Use basicConfig for initial setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        # Force setup even if already configured by other imports
        force=True
    )
    # Get logger instance *after* basicConfig
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging setup complete. Log file: {log_file}")
    return log_file


def set_seed(seed: int) -> None:
    """Sets random seeds for torch and numpy for reproducibility.

    Args:
        seed: The seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: For full determinism on CUDA (can slow down)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    logger.info(f"🌱 Set random seed to {seed}")


def get_device() -> torch.device:
    """Determines and returns the optimal torch device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        logger.info("🚀 Using Apple Silicon MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("🐌 Using CPU device.")
    return device


def get_data_loaders(batch_size: int, data_dir='./model/data') -> tuple:
    """Creates MNIST train/test DataLoaders with appropriate transforms.

    Adjusts num_workers and pin_memory for MPS compatibility.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Directory for MNIST data.

    Returns:
        Tuple containing (train_loader, test_loader).
    """
    num_workers = 4
    pin_memory = True
    # Adjust based on the actual device being used for training
    current_device_type = get_device().type
    if current_device_type == 'mps':
        num_workers = 0  # Recommended for MPS stability
        pin_memory = False # Recommended for MPS
        logger.info("⚙️ Adjusted DataLoader for MPS: num_workers=0, "
                     "pin_memory=False.")
    elif current_device_type == 'cpu':
        pin_memory = False # pin_memory only benefits GPU copies

    train_transform = get_train_transforms() # Uses strong augmentation
    test_transform = get_test_transforms()

    try:
        logger.info(f"💾 Loading MNIST dataset from/to: {data_dir}")
        os.makedirs(data_dir, exist_ok=True) # Ensure directory exists
        train_dataset = datasets.MNIST(root=data_dir, train=True,
                                       download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      download=True, transform=test_transform)
        logger.info("✅ MNIST dataset loaded.")
    except Exception as e:
        logger.critical(f"🔥 Failed to load MNIST dataset from {data_dir}: {e}")
        raise  # Re-raise critical error

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f"📦 DataLoaders created with batch size {batch_size}.")
    return train_loader, test_loader


def train_model(num_epochs: int, batch_size: int, learning_rate: float,
                device: torch.device, save_path: str, log_dir: str) -> tuple:
    """Trains the MNIST classifier, saves the best model, returns path/history.

    Args:
        num_epochs: Number of epochs to train.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        device: Device for training (mps, cuda, cpu).
        save_path: Path to save the best performing model checkpoint.
        log_dir: Directory to save training history JSON.

    Returns:
        Tuple: (path_to_best_model, training_history_dict).
    """
    train_start_time = time.time()
    logger.info(f"🏋️ Starting training for {num_epochs} epochs...")

    # --- Setup ---
    try:
        train_loader, test_loader = get_data_loaders(batch_size)
    except Exception:
        logger.critical("🔥 Aborting training due to DataLoader failure.")
        # Return dummy values or re-raise
        return None, {}

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=False # Less verbose
    )
    logger.info("✅ Model, Loss, Optimizer, Scheduler initialized.")

    # --- Training Loop ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} '
                          '[Train]', leave=False)
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

            current_loss = loss.item()
            current_acc = correct_train / total_train if total_train > 0 else 0.0
            train_pbar.set_postfix({'Loss': f"{current_loss:.4f}",
                                    'Acc': f"{current_acc:.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        # Validation
        model.eval()
        # Use context manager for evaluation if evaluate_model supports it
        # or ensure evaluate_model uses torch.no_grad() internally
        epoch_val_loss, epoch_val_acc = evaluate_model(
            model, test_loader, criterion, device
        )

        # Scheduler Step + Logging LR change
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
             logger.info(f"📉 Learning rate reduced to {new_lr:.6f} "
                         f"at epoch {epoch+1}")

        epoch_duration = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"T_Loss={epoch_train_loss:.4f} | T_Acc={epoch_train_acc:.4f} | "
            f"V_Loss={epoch_val_loss:.4f} | V_Acc={epoch_val_acc:.4f} | "
            f"🕒 {epoch_duration:.2f}s"
        )

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                torch.save(model.state_dict(), save_path)
                logger.info(
                    f"💾 Epoch {best_epoch}: New best model saved "
                    f"(Val Loss: {best_val_loss:.4f})"
                )
            except Exception as e_save:
                logger.error(f"🔥 Error saving model at epoch {epoch+1}: "
                             f"{e_save}")

    # --- Post-Training ---
    total_training_time = time.time() - train_start_time
    logger.info(f"🏁 Training finished in {total_training_time:.2f} seconds.")
    if best_epoch != -1:
        logger.info(f"🏆 Best model from Epoch {best_epoch} saved to "
                     f"{save_path}")
    else:
        logger.warning("⚠️ No best model was saved (validation loss might "
                       "not have improved).")

    # Plot History
    history_plot_path = os.path.join(OUTPUT_FIG_DIR, HISTORY_PLOT_FILENAME)
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    try:
        # Ensure plot_training_history accepts save_path argument
        plot_training_history(history, save_path=history_plot_path)
        logger.info(f"📈 Training history plot saved to {history_plot_path}")
    except TypeError as e_plot:
        logger.warning(f"⚠️ Plotting failed (check function signature): {e_plot}")
    except Exception as e_plot_other:
        logger.warning(f"⚠️ Could not plot training history: {e_plot_other}")


    # Save History Data
    history_file_name = f"training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    history_file_path = os.path.join(log_dir, history_file_name)
    try:
        with open(history_file_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"📜 Training history data saved to {history_file_path}")
    except Exception as e_hist:
        logger.warning(f"⚠️ Could not save training history JSON: {e_hist}")

    return save_path if best_epoch != -1 else None, history


def final_evaluation(model_path: str, device: torch.device,
                     batch_size: int) -> None:
    """Performs final evaluation on the test set using the best model.

    Generates reports, saves debug images, and plots reliability diagram.

    Args:
        model_path: Path to the saved best model state_dict.
        device: Device for evaluation (mps, cuda, cpu).
        batch_size: Batch size for the test DataLoader.
    """
    if not model_path or not os.path.exists(model_path):
         logger.error("🔥 Final Evaluation Failed: Invalid or missing "
                      f"model path '{model_path}'")
         return

    logger.info("\n" + "="*30 + "\n🔬 Performing Final Evaluation 🔬\n" + "="*30)

    # Create Test Loader for final evaluation
    try:
        _, test_loader = get_data_loaders(batch_size)
    except Exception:
        logger.error("🔥 Final Evaluation Failed: Cannot create test loader.")
        return

    model = MNISTClassifier().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"✅ Loaded best model from {model_path} for final eval.")
    except Exception as e:
        logger.error(f"🔥 Failed to load best model for final evaluation: {e}")
        return

    # --- Save Debug Test Images ---
    try:
        os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
        dataiter = iter(test_loader)
        images, _ = next(dataiter)
        img_to_save = images[0].clone().cpu() # Process on CPU

        # Save Normalized Version
        norm_save_path = os.path.join(OUTPUT_DEBUG_DIR,
                                      DEBUG_TEST_NORM_FILENAME)
        torchvision.utils.save_image(img_to_save, norm_save_path)
        logger.info(f"📸 Saved example normalized test input to "
                    f"{norm_save_path}")

        # Save Un-normalized Version
        mean = torch.tensor(MNIST_MEAN).view(-1, 1, 1)
        std = torch.tensor(MNIST_STD).view(-1, 1, 1)
        img_unnorm = img_to_save * std + mean
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        unnorm_save_path = os.path.join(OUTPUT_DEBUG_DIR,
                                        DEBUG_TEST_UNNORM_FILENAME)
        torchvision.utils.save_image(img_unnorm, unnorm_save_path)
        logger.info(f"🖼️ Saved example un-normalized test input to "
                    f"{unnorm_save_path}")
    except Exception as e_save:
        logger.error(f"⚠️ Failed to save debug input images: {e_save}")

    # --- Generate Classification Report & Confusion Matrix ---
    cm_path = os.path.join(OUTPUT_FIG_DIR, FINAL_CM_FILENAME)
    report_path = FINAL_REPORT_FILENAME
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    try:
        logger.info("⏳ Generating classification report & CM...")
        generate_evaluation_report(
            model=model, data_loader=test_loader, device=device,
            save_cm_path=cm_path, save_report_path=report_path
        )
        logger.info(f"📊 Classification reports saved: {cm_path}, {report_path}")
    except Exception as e_eval:
        logger.error(f"🔥 Failed during report generation: {e_eval}")

    # --- Generate Reliability Diagram (Before Calibration) ---
    reliability_plot_path = os.path.join(OUTPUT_FIG_DIR,
                                         RELIABILITY_PLOT_BEFORE_FILENAME)
    try:
        logger.info("⏳ Generating reliability diagram (before calibration)...")
        # 1. Get probabilities and labels from the test set
        confidences, predictions, true_labels = get_probabilities_labels(
            model=model, data_loader=test_loader, device=device
        )
        # 2. Plot the diagram
        if confidences is not None: # Check if data was obtained
            ece = plot_reliability_diagram(
                confidences=confidences,
                predictions=predictions,
                true_labels=true_labels,
                num_bins=15, # Use 15 bins for finer granularity
                save_path=reliability_plot_path
            )
            logger.info(f"📉 Reliability diagram saved to "
                        f"{reliability_plot_path} | ECE={ece:.4f}")
        else:
             logger.warning("⚠️ Skipping reliability diagram: failed to get "
                            "probabilities/labels.")
    except Exception as e_calib:
        logger.error(f"🔥 Failed during reliability diagram generation: "
                     f"{e_calib}", exc_info=True) # Log traceback too
    # -------------------------------------------------------


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f"Num epochs (default: {NUM_EPOCHS})")
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

    log_file_path = setup_logging(args.log_dir)
    set_seed(args.seed)
    selected_device = get_device()

    # Train
    best_model_file_path, _ = train_model( # Ignore history return for now
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=selected_device,
        save_path=args.save_path,
        log_dir=args.log_dir
    )

    # Evaluate (includes reliability plot now)
    final_evaluation(
        model_path=best_model_file_path,
        device=selected_device,
        batch_size=args.batch_size
    )

    logger.info("✅ Training script finished.")