#!/bin/bash
# MNIST Digit Classifier
# Copyright (c) 2025
# File: setup.sh
# Description: Setup script for MNIST Excellence Project
# Created: 2025-03-26
# Updated: 2025-03-26

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo ""
echo "=================================================="
echo -e "${GREEN}🚀 MNIST EXCELLENCE PROJECT - SETUP 🚀${NC}"
echo "=================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${BLUE}📌 Detected Python version:${NC} $PYTHON_VERSION"
if [[ $PYTHON_VERSION != *"Python 3."* ]]; then
    echo -e "${RED}❌ Error: Python 3 is required!${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${BLUE}📋 Creating virtual environment...${NC}"
if [[ -d "venv" ]]; then
    echo -e "${YELLOW}⚠️ Virtual environment already exists!${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}📌 Using existing virtual environment.${NC}"
    else
        echo -e "${BLUE}📌 Removing existing virtual environment...${NC}"
        rm -rf venv
        python -m venv venv
        echo -e "${GREEN}✅ Virtual environment created!${NC}"
    fi
else
    python -m venv venv
    echo -e "${GREEN}✅ Virtual environment created!${NC}"
fi

# Activate virtual environment
echo -e "\n${BLUE}📋 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated!${NC}"

# Install dependencies
echo -e "\n${BLUE}📋 Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed!${NC}"

# Create directory structure
echo -e "\n${BLUE}📋 Creating directory structure...${NC}"
mkdir -p data
mkdir -p saved_models
mkdir -p outputs/logs
mkdir -p outputs/figures
mkdir -p utils
mkdir -p model
echo -e "${GREEN}✅ Directory structure created!${NC}"

# Set up environment variables
echo -e "\n${BLUE}📋 Setting up environment variables...${NC}"
cat > .env << EOF
# Environment variables for MNIST Excellence Project
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
PYTORCH_ENABLE_MPS_FALLBACK=1
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4
NUMEXPR_NUM_THREADS=4
EOF

echo -e "${GREEN}✅ Environment variables set up!${NC}"

# Create environment activation script
echo -e "\n${BLUE}📋 Creating environment activation script...${NC}"
cat > activate_env.sh << EOF
#!/bin/bash
# Activate environment for MNIST Excellence Project
source venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
echo "🚀 MNIST Excellence Project environment activated!"
EOF

chmod +x activate_env.sh
echo -e "${GREEN}✅ Environment activation script created!${NC}"

# Run MPS verification
echo -e "\n${BLUE}📋 Would you like to run MPS verification now? (y/n)${NC}"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Make the verification script executable
    chmod +x run_mps_verification.sh
    
    # Run the verification script
    ./run_mps_verification.sh
fi

# Set up device strategy
echo -e "\n${BLUE}📋 Setting up initial device strategy...${NC}"
python -c "
import os
import json
import torch

os.makedirs('outputs', exist_ok=True)

device_strategy = {
    'training_device': 'cpu',
    'inference_device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'batch_size_train': 64,
    'batch_size_inference': 256,
    'mixed_precision': False
}

with open('outputs/device_strategy.json', 'w') as f:
    json.dump(device_strategy, f, indent=4)

print('✅ Device strategy created at outputs/device_strategy.json')
"

# Print completion message
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}✅ Setup completed successfully!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "\n${BLUE}📋 Next steps:${NC}"
echo "1. Activate the environment: source activate_env.sh"
echo "2. Start with Day 1 tasks: python utils/environment_setup.py"
echo "3. Generate synthetic data: python utils/augmentation.py"
echo ""
echo -e "${YELLOW}Happy coding! 🚀${NC}"
echo ""