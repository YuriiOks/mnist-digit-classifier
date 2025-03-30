# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: scripts/test_evaluation_utils.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-27
# Updated: 2025-03-30

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm if not already in evaluation.py's imports

# --- Configuration ---
MODEL_PATH = "model/saved_models/mnist_classifier.pt"
DATA_DIR = "./data"
BATCH_SIZE = 128  # Use a reasonable batch size for evaluation
OUTPUT_CM_PATH = "test_confusion_matrix.png"
OUTPUT_REPORT_PATH = "test_classification_report.txt"

# --- Add project root to path ---
# This allows importing from model and utils directories
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Imports from your project ---
try:
    from model.model import MNISTClassifier

    # Import the specific function we want to test
    from model.utils.evaluation import generate_evaluation_report

    # Make sure necessary functions used by generate_evaluation_report are importable
    # (like get_predictions_and_labels, plot_confusion_matrix)
    # Also ensure evaluation.py imports its own needs (like tqdm, plt, sns, etc.)
except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    print("Ensure you are running this script from the 'scripts' directory")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def main():
    """
    Main function to test evaluation utilities.
    This function performs the following steps:
    1. Set the device (CPU or GPU).
    2. Load the MNIST model.
    3. Load the MNIST test dataset.
    """
    print("🚀 Starting Evaluation Utilities Test 🚀")

    # --- 1. Set Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("💻 Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("💻 Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU device.")

    # --- 2. Load Model ---
    print(f"💾 Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("👉 Please train a model first or create a dummy one for testing.")
        # Option: Create dummy model here if needed
        # print("Creating dummy model...")
        # model = MNISTClassifier()
        # os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # torch.save(model.state_dict(), MODEL_PATH)
        # print(f"Dummy model saved to {MODEL_PATH}")
        # model = model.to(device) # Move dummy model to device
        sys.exit(1)  # Exit if no model
    else:
        model = MNISTClassifier().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()  # Set to evaluation mode
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model state_dict: {e}")
            sys.exit(1)

    # --- 3. Load Data ---
    print("📦 Loading MNIST test data...")
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
        ]
    )
    try:
        test_dataset = datasets.MNIST(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=test_transform,
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"✅ Test data loaded ({len(test_dataset)} samples).")
    except Exception as e:
        print(f"❌ Error loading MNIST data: {e}")
        sys.exit(1)

    # --- 4. Run Evaluation ---
    print("\n🧪 Calling generate_evaluation_report...")
    try:
        # Call the function to test
        cm, report = generate_evaluation_report(
            model=model,
            data_loader=test_loader,
            device=device,
            save_cm_path=OUTPUT_CM_PATH,
            save_report_path=OUTPUT_REPORT_PATH,
        )
        print("\n✅ generate_evaluation_report finished.")

        # --- 5. Verify Outputs ---
        print("\n🧐 Verifying output files...")
        success = True
        if os.path.exists(OUTPUT_CM_PATH):
            print(f"✔️ Confusion Matrix plot found: {OUTPUT_CM_PATH}")
        else:
            print(f"❌ Confusion Matrix plot NOT found: {OUTPUT_CM_PATH}")
            success = False

        if os.path.exists(OUTPUT_REPORT_PATH):
            print(f"✔️ Classification Report file found: {OUTPUT_REPORT_PATH}")
            # Optional: check if file has content
            if os.path.getsize(OUTPUT_REPORT_PATH) > 0:
                print("    File is not empty.")
            else:
                print("    ⚠️ Warning: Report file is empty.")
                # success = False # Decide if empty file is failure
        else:
            print(f"❌ Classification Report file NOT found: {OUTPUT_REPORT_PATH}")
            success = False

        if success:
            print(
                "\n🎉 Test Passed: Evaluation functions seem to work and generated output files."
            )
        else:
            print(
                "\n🔥 Test Failed: Check errors above and ensure functions generate outputs."
            )

    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback


if __name__ == "__main__":
    main()
