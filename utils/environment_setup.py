# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/environment_setup.py
# Description: Environment setup for optimal M4 Pro performance
# Created: 2025-03-26
# Updated: 2025-03-26

import os
import torch
import numpy as np
import random
import json
import logging
import platform
from utils.device_strategy import determine_optimal_devices, set_environment_variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/logs/setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_system_info():
    """Print system information for debugging."""
    logger.info("🖥️ System Information:")
    logger.info(f"  - OS: {platform.system()} {platform.release()}")
    logger.info(f"  - Python: {platform.python_version()}")
    logger.info(f"  - PyTorch: {torch.__version__}")
    logger.info(f"  - CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  - MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"  - CPU count: {os.cpu_count()}")
    
    # Check MPS device if available
    if torch.backends.mps.is_available():
        logger.info("  - MPS device: Available")
        
        # Check if MPS is built and available
        logger.info(f"  - MPS built: {torch.backends.mps.is_built()}")

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"🎲 Random seed set to {seed}")

def create_directory_structure():
    """Create necessary directories."""
    # List of directories to create
    directories = [
        "data",
        "saved_models",
        "outputs",
        "outputs/figures",
        "outputs/logs",
        "utils",
        "model"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"📁 Created directory structure")

def setup_environment(seed=42):
    """
    Set up the environment for optimal performance.
    
    Args:
        seed: Random seed for reproducibility
    """
    logger.info("🚀 Setting up environment...")
    
    # Print system information
    print_system_info()
    
    # Set random seed
    set_random_seed(seed)
    
    # Create directory structure
    create_directory_structure()
    
    # Set environment variables for optimal performance
    set_environment_variables()
    
    # Determine optimal device strategy
    strategy = determine_optimal_devices()
    
    logger.info("✅ Environment setup complete!")
    
    return strategy

if __name__ == "__main__":
    # Run setup when script is executed directly
    setup_environment()