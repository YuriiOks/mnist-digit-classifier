import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
import time
import datetime
import json
from model import MNISTClassifier
from utils.augmentation import get_train_transforms, get_test_transforms
from utils.evaluation import evaluate_model, plot_training_history

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
SAVE_PATH = "saved_models/mnist_classifier.pt"
LOG_DIR = "logs"

def setup_logging(log_dir=LOG_DIR):
    """Set up logging to file with timestamps."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging set up. Log file: {log_file}")
    return log_file

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def get_data_loaders(batch_size, num_workers=4):
    """
    Create train and test data loaders with appropriate transforms.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get transforms for training (with augmentation) and testing
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Load training dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_model():
    """Train the MNIST classifier model."""
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting model training")
    
    # Record start time
    start_time = time.time()
    
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    logging.info(f"Random seed set to {RANDOM_SEED}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    
    if device.type == "cuda":
        logging.info("Using CUDA GPU")
    elif device.type == "mps":
        logging.info("Using Apple Silicon MPS (Metal Performance Shaders)")
    else:
        logging.info(f"Using CPU with {torch.get_num_threads()} threads for computation")
    
    # Get data loaders
    num_workers = 4
    logging.info(f"Using {num_workers} workers for data loading")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE, num_workers)
    logging.info(f"Data loaders created with batch size: {BATCH_SIZE}")
    
    # Initialize model
    model = MNISTClassifier().to(device)
    logging.info(f"Model initialized on {device}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    logging.info(f"Optimizer initialized with learning rate: {LEARNING_RATE}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for images, labels in train_pbar:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate training statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        # Calculate epoch training statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        logging.info("Starting validation phase")
        model.eval()
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr != prev_lr:
            logging.info(f"Learning rate adjusted from {prev_lr} to {curr_lr}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            logging.info(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.2f} seconds:')
        logging.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logging.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Plot training history
    logging.info("Training completed. Generating training history plot")
    plot_training_history(history)
    
    # Save history to file for later analysis
    history_file = os.path.join(LOG_DIR, f"training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_file, 'w') as f:
        json.dump(history, f)
    logging.info(f"Training history saved to {history_file}")
    
    # Final evaluation on test set
    logging.info("Performing final evaluation with best model")
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    logging.info(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Log total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST classifier")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    args = parser.parse_args()
    
    # Update parameters if provided
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    train_model()
