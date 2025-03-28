# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/train.py
# Description: Script for training the MNIST digit classifier model.
# Created: Earlier Date
# Updated: 2025-03-28 (Integrated calibration steps, cleaned up)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
import time
import datetime
import json
import sys

# Project-specific imports
try:
    from model.model import MNISTClassifier
    from model.utils.augmentation import get_train_transforms, get_test_transforms
    from model.utils.evaluation import (
        evaluate_model, plot_training_history, generate_evaluation_report
    )
    from model.utils.calibration import (
        ModelWithTemperature, plot_reliability_diagram, get_probabilities_labels
    )
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported project modules.")
except ImportError as e:
    print(f"🔥 CRITICAL ERROR importing project modules: {e}")
    print("Ensure script is run relative to project root or PYTHONPATH is set.")
    sys.exit(1)

# --- Constants ---
RANDOM_SEED = 42
BATCH_SIZE = 512
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1 # Use 10% of training data for validation/temp tuning
SAVE_PATH = "model/saved_models/mnist_classifier.pt"
OPTIMAL_TEMP_FILE = "model/saved_models/optimal_temperature.json"
LOG_DIR = "model/logs"
OUTPUT_FIG_DIR = "outputs/figures"
OUTPUT_DEBUG_DIR = "outputs/debug_images"
HISTORY_PLOT_FILENAME = "training_history.png"
FINAL_CM_FILENAME = "final_confusion_matrix.png"
FINAL_REPORT_FILENAME = "final_classification_report.txt"
RELIABILITY_PLOT_BEFORE_FILENAME = "reliability_diagram_before_calib.png"
RELIABILITY_PLOT_AFTER_FILENAME = "reliability_diagram_after_calib.png"
DEBUG_TEST_NORM_FILENAME = "dbg_test_input_normalized.png"
DEBUG_TEST_UNNORM_FILENAME = "dbg_test_input_unnormalized.png"
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
# -----------------

def setup_logging(log_dir=LOG_DIR) -> str:
    """Sets up logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True
    )
    logger = logging.getLogger(__name__) # Re-get after basicConfig
    logger.info(f"📝 Logging setup complete. Log file: {log_file}")
    return log_file

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info(f"🌱 Set random seed to {seed}")

def get_device() -> torch.device:
    """Determines best available device."""
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

def get_data_loaders(batch_size: int, data_dir='./model/data',
                     val_split=VALIDATION_SPLIT, seed=RANDOM_SEED) -> tuple:
    """Creates MNIST train, validation, and test DataLoaders."""
    num_workers = 0 if get_device().type == 'mps' else 4
    pin_memory = False if get_device().type != 'cuda' else True
    if get_device().type == 'mps': logger.info(
        "⚙️ Adjusted DataLoader for MPS: num_workers=0, pin_memory=False.")

    train_transform = get_train_transforms()
    test_transform = get_test_transforms() # Validation uses test transforms

    try:
        logger.info(f"💾 Loading MNIST dataset from/to: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        # Load full training set with train transforms initially
        full_train_dataset = datasets.MNIST(root=data_dir, train=True,
                                        download=True, transform=train_transform)
        # Load test set with test transforms
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      download=True, transform=test_transform)
        logger.info("✅ MNIST datasets loaded.")

        # Split training data
        num_train = len(full_train_dataset)
        num_val = int(val_split * num_train)
        if num_val == 0 and val_split > 0 and num_train > 0:
             num_val = 1 # Ensure at least one validation sample if requested
        num_train -= num_val
        if num_train <= 0 or num_val <= 0:
             raise ValueError("Train/Validation split resulted in zero samples "
                              "for one set.")

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset_indices_only = random_split(
            full_train_dataset, [num_train, num_val], generator=generator
        )

        # Create validation dataset VIEW with test transforms applied
        # Important: Access original dataset for validation transforms
        val_dataset_view = datasets.MNIST(root=data_dir, train=True,
                                          download=False, # Already downloaded
                                          transform=test_transform)
        val_subset_final = torch.utils.data.Subset(
            val_dataset_view, val_subset_indices_only.indices
        )
        logger.info(f"📊 Split training data: {len(train_subset)} train, "
                     f"{len(val_subset_final)} validation.")

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset_final, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

        logger.info(f"📦 DataLoaders created (Train, Validation, Test).")
        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.critical(f"🔥 Failed to load/split dataset: {e}", exc_info=True)
        raise

def train_model(num_epochs: int, batch_size: int, learning_rate: float,
                device: torch.device, save_path: str, log_dir: str) -> tuple:
    """Trains model, saves best based on validation loss, returns path/history."""
    train_start_time = time.time()
    logger.info(f"🏋️ Starting training for {num_epochs} epochs...")

    try:
        train_loader, val_loader, _ = get_data_loaders(batch_size) # Use val_loader
    except Exception:
        logger.critical("🔥 Aborting: DataLoader failure.")
        return None, {}

    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=False
    )
    logger.info("✅ Model setup complete.")

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
            # Training step logic...
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Accumulate stats...
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            # Update tqdm...
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}",
                                    'Acc': f"{correct_train/total_train:.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        # Validation using val_loader
        model.eval()
        epoch_val_loss, epoch_val_acc = evaluate_model(
            model, val_loader, criterion, device # Use val_loader now
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)
        if optimizer.param_groups[0]['lr'] < current_lr:
             logger.info(f"📉 LR reduced to {optimizer.param_groups[0]['lr']:.6f}")

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

        # Save Best Model based on Validation Loss
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
                logger.error(f"🔥 Error saving model: {e_save}")

    total_training_time = time.time() - train_start_time
    logger.info(f"🏁 Training finished in {total_training_time:.2f} seconds.")
    if best_epoch != -1: logger.info(f"🏆 Best model from Epoch {best_epoch}")
    else: logger.warning("⚠️ No best model saved.")

    # Plot History
    history_plot_path = os.path.join(OUTPUT_FIG_DIR, HISTORY_PLOT_FILENAME)
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    try:
        plot_training_history(history, save_path=history_plot_path)
        logger.info(f"📈 History plot saved: {history_plot_path}")
    except Exception as e_plot:
        logger.warning(f"⚠️ Failed to plot training history: {e_plot}")

    # Save History Data
    history_file_name = f"train_hist_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    history_file_path = os.path.join(log_dir, history_file_name)
    try:
        with open(history_file_path, 'w') as f: json.dump(history, f, indent=4)
        logger.info(f"📜 History data saved: {history_file_path}")
    except Exception as e_hist:
        logger.warning(f"⚠️ Failed to save history JSON: {e_hist}")

    return save_path if best_epoch != -1 else None, history


def optimize_temperature(model_path: str, device: torch.device,
                         batch_size: int) -> float:
    """Loads best model, tunes temperature on validation set, saves T."""
    optimal_temp = 1.0 # Default temperature
    if not model_path or not os.path.exists(model_path):
         logger.error("🔥 Temp Opt Failed: Invalid model path.")
         return optimal_temp

    logger.info("\n" + "="*30 + "\n🌡️ Optimizing Temperature \n" + "="*30)
    try:
        # Get validation loader (only needs test transforms)
        _, val_loader, _ = get_data_loaders(batch_size)
    except Exception:
        logger.error("🔥 Temp Opt Failed: Cannot create val loader.")
        return optimal_temp

    base_model = MNISTClassifier() # Instantiate on CPU first
    try:
        base_model.load_state_dict(torch.load(model_path,
                                               map_location='cpu'))
        logger.info(f"✅ Loaded base model from {model_path}")
    except Exception as e:
        logger.error(f"🔥 Failed to load base model for temp scaling: {e}")
        return optimal_temp

    # Optimize
    temp_model = ModelWithTemperature(base_model)
    try:
        optimal_temp = temp_model.set_temperature(val_loader, device)
    except Exception as e_tune:
         logger.error(f"🔥 Error during temperature tuning: {e_tune}",
                      exc_info=True)
         optimal_temp = temp_model.temperature.item() # Use current T

    # Save optimal temperature
    try:
        os.makedirs(os.path.dirname(OPTIMAL_TEMP_FILE), exist_ok=True)
        with open(OPTIMAL_TEMP_FILE, 'w') as f:
            json.dump({"temperature": optimal_temp}, f)
        logger.info(f"💾 Optimal temperature {optimal_temp:.4f} saved to "
                    f"{OPTIMAL_TEMP_FILE}")
    except Exception as e_save_t:
        logger.error(f"🔥 Failed to save optimal temperature: {e_save_t}")

    return optimal_temp


def final_evaluation(model_path: str, device: torch.device,
                     batch_size: int, temperature: float = 1.0) -> None:
    """Performs final eval: reports, debug images, reliability diagram."""
    if not model_path or not os.path.exists(model_path):
         logger.error("🔥 Final Eval Failed: Invalid model path.")
         return
    logger.info("\n" + "="*30 + "\n🔬 Performing Final Evaluation \n" + "="*30)

    try:
        _, _, test_loader = get_data_loaders(batch_size) # Get test loader
    except Exception:
        logger.error("🔥 Final Eval Failed: Cannot create test loader.")
        return

    model = MNISTClassifier().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"✅ Loaded best model from {model_path}")
    except Exception as e:
        logger.error(f"🔥 Failed to load model for final eval: {e}")
        return

    # Save Debug Images
    try:
        os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
        images, _ = next(iter(test_loader))
        img_norm = images[0].clone().cpu()
        norm_path = os.path.join(OUTPUT_DEBUG_DIR, DEBUG_TEST_NORM_FILENAME)
        torchvision.utils.save_image(img_norm, norm_path)
        logger.info(f"📸 Saved norm test input: {norm_path}")
        mean=torch.tensor(MNIST_MEAN).view(-1,1,1); std=torch.tensor(MNIST_STD).view(-1,1,1)
        img_unnorm = torch.clamp(img_norm * std + mean, 0, 1)
        unnorm_path = os.path.join(OUTPUT_DEBUG_DIR, DEBUG_TEST_UNNORM_FILENAME)
        torchvision.utils.save_image(img_unnorm, unnorm_path)
        logger.info(f"🖼️ Saved unnorm test input: {unnorm_path}")
    except Exception as e_save: logger.error(f"⚠️ Failed save debug imgs: {e_save}")

    # Generate Classification Report & CM
    cm_path = os.path.join(OUTPUT_FIG_DIR, FINAL_CM_FILENAME)
    report_path = FINAL_REPORT_FILENAME
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    try:
        logger.info("⏳ Generating classification report & CM...")
        generate_evaluation_report(model=model, data_loader=test_loader,
            device=device, save_cm_path=cm_path, save_report_path=report_path)
        logger.info(f"📊 Reports saved: {cm_path}, {report_path}")
    except Exception as e: logger.error(f"🔥 Failed report gen: {e}")

    # Generate Reliability Diagram (using scaled logits)
    reliability_plot_path = os.path.join(
        OUTPUT_FIG_DIR,
        f"reliability_diagram_T_{temperature:.2f}.png"
    )
    try:
        logger.info(f"⏳ Generating reliability diagram (T={temperature:.3f})...")
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Reliability Data"):
                all_logits.append(model(images.to(device)))
                all_labels.append(labels) # Keep labels on CPU
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).numpy()

        # Apply scaling and get probabilities
        scaled_logits = all_logits / max(temperature, 1e-6) # Ensure T > 0
        probabilities = F.softmax(scaled_logits, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        confidences_np = confidences.cpu().numpy()
        predictions_np = predictions.cpu().numpy()

        if confidences_np is not None:
            ece = plot_reliability_diagram(
                confidences=confidences_np, predictions=predictions_np,
                true_labels=all_labels, num_bins=15,
                save_path=reliability_plot_path
            )
            logger.info(f"📉 Reliability diagram saved: {reliability_plot_path}"
                        f" | ECE={ece:.4f}")
        else: logger.warning("⚠️ Skipping reliability plot: no data.")
    except Exception as e: logger.error(f"🔥 Failed reliability gen: {e}",
                                         exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--save_path', type=str, default=SAVE_PATH)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    setup_logging(args.log_dir)
    set_seed(args.seed)
    selected_device = get_device()

    best_model_path, _ = train_model(
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, device=selected_device,
        save_path=args.save_path, log_dir=args.log_dir
    )

    optimal_T = optimize_temperature(
        model_path=best_model_path, device=selected_device,
        batch_size=args.batch_size
    )

    final_evaluation(
        model_path=best_model_path, device=selected_device,
        batch_size=args.batch_size, temperature=optimal_T
    )

    logger.info("✅ Training & Calibration script finished.")