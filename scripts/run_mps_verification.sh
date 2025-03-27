#!/bin/bash
# MNIST Digit Classifier
# Copyright (c) 2025
# File: run_mps_verification.sh
# Description: Script to run MPS verification tests
# Created: 2025-03-26
# Updated: 2025-03-26

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo ""
echo "=================================================="
echo -e "${GREEN}🚀 MNIST EXCELLENCE PROJECT - MPS VERIFICATION 🚀${NC}"
echo "=================================================="
echo ""

# Set environment variables for optimal MPS performance
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use maximum available memory
export PYTORCH_ENABLE_MPS_FALLBACK=1         # Fallback for unsupported operations

# Print environment variables
echo -e "${BLUE}📚 Library Versions:${NC}"
echo "-----------------"
echo -e "🐍 Python: $(python --version)"
echo -e "🔥 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo -e "🖼️  TorchVision: $(python -c 'import torch; print(torch.version.cuda if hasattr(torch.version, "cuda") else "None")')"
echo -e "🔢 NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo -e "📊 Matplotlib: $(python -c 'import matplotlib; print(matplotlib.__version__)')"
echo "-----------------"
echo ""

# Run MPS verification
echo -e "${BLUE}🔍 Running MPS verification...${NC}"
echo ""
python utils/mps_verification.py

# Print next steps
echo ""
echo "=============================================="
echo -e "${YELLOW}🎯 NEXT STEPS${NC}"
echo "=============================================="
echo ""
echo "Based on the MPS verification results:"
echo ""
echo "1. Use the recommended batch size for training"
echo "2. Set the environment variables as suggested"
echo "3. Continue to the next part of the plan:"
echo "   - Setting up the evaluation framework"
echo "   - Implementing confusion matrix analysis"
echo ""
echo "=============================================="