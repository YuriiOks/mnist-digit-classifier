# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: model/app.py
# Description: Flask API for serving the MNIST classifier model.
# Created: 2025-03-06
# Updated: 2025-03-30

import os
import base64
import io
import json
import sys
import inspect
import logging
import time
from typing import Optional

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Add /app to sys.path for Docker compatibility ---
# (Useful when running with `python app.py` in WORKDIR /app)
if "/app" not in sys.path:
    sys.path.insert(0, "/app")
    logger.info("Manually added /app to sys.path")

# --- Import Model and Preprocessing Utils ---
try:
    # Assumes model.py is directly in /app or PYTHONPATH allows finding it
    from model import MNISTClassifier

    logger.info("✅ Imported MNISTClassifier from model.py")
except ImportError as e:
    logger.critical(f"🔥 CRITICAL ERROR importing MNISTClassifier: {e}")
    sys.exit(1)

try:
    # Assumes utils/preprocessing.py is under /app
    from utils.preprocessing import center_digit

    logger.info("✅ Imported center_digit from utils.preprocessing")
except ImportError as e_imp_utils:
    logger.error(f"🔥 Failed import center_digit: {e_imp_utils}. Centering disabled.")
    center_digit = None
except AttributeError as e_attr_center:
    logger.error(f"🔥 Failed to find center_digit function: {e_attr_center}")
    center_digit = None
# --------------------------------------------

app = Flask(__name__)

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))  # /app in container
model_dir = os.path.join(script_dir, "saved_models")  # /app/saved_models
default_model_path = os.path.join(model_dir, "mnist_classifier.pt")
temp_path = os.path.join(model_dir, "optimal_temperature.json")
model_path = os.environ.get("MODEL_PATH", default_model_path)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEBUG_IMG_DIR = "outputs/debug_images"  # Relative to /app
# ---------------------

# --- Load Model ---
model_instance = MNISTClassifier()
MODEL_LOADED = False
try:
    logger.info(f"💾 Attempting to load model from: {model_path}")
    model_instance.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model_instance.eval()
    MODEL_LOADED = True
    logger.info(f"✅ Loaded model successfully from {model_path}")
except FileNotFoundError:
    logger.error(f"🔥 ERROR: Model file not found at: {model_path}")
except Exception as e:
    logger.error(f"⚠️ Could not load model state_dict: {e}")
# ------------------

# --- Load Temperature ---
TEMPERATURE = 1.0
if MODEL_LOADED:
    try:
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                data = json.load(f)
                temp_val = data.get("temperature")
                if isinstance(temp_val, (int, float)) and temp_val > 1e-6:
                    TEMPERATURE = float(temp_val)
                    logger.info(f"🌡️ Loaded optimal temperature: {TEMPERATURE:.4f}")
                else:
                    logger.warning(f"⚠️ Invalid temp value. Using T=1.0.")
        else:
            logger.warning(f"⚠️ Temp file {temp_path} not found. Using T=1.0.")
    except Exception as e:
        logger.error(f"🔥 Error loading temp: {e}. Using T=1.0.")
else:
    logger.warning("⚠️ Model not loaded, skipping temp load.")
# -----------------------


# --- Image Preprocessing Function ---
def preprocess_image(base64_image_data: str) -> Optional[torch.Tensor]:
    """Decodes, ensures W-on-B, pads, centers, preprocesses, saves debug."""
    try:
        # 1. Decode base64 -> PIL Image
        if "," in base64_image_data:
            _, encoded = base64_image_data.split(",", 1)
        else:
            encoded = base64_image_data
        image_bytes = base64.b64decode(encoded)
        image_initial = Image.open(io.BytesIO(image_bytes)).convert("L")

        # 2. Ensure WHITE Digit on BLACK Background using Mean Check
        img_array = np.array(image_initial)
        mean_intensity = np.mean(img_array)
        logger.debug(f"🎨 Initial mean intensity: {mean_intensity:.2f}")
        if mean_intensity >= 128:  # Input is Black-on-White
            logger.debug("🎨 Inverting Black-on-White input -> White-on-Black.")
            image_w_on_b = ImageOps.invert(image_initial)
            padding_fill_color = 0  # Black padding
        else:  # Input is already White-on-Black
            logger.debug("🎨 Input is already White-on-Black. No inversion.")
            image_w_on_b = image_initial
            padding_fill_color = 0  # Black padding

        # 3. Add Padding (Black)
        try:
            padding = 30
            image_padded = ImageOps.expand(
                image_w_on_b, border=padding, fill=padding_fill_color
            )
            logger.debug(f"✅ Added {padding}px padding (black).")
        except Exception as e:
            logger.error(f"⚠️ Padding failed: {e}. Using unpadded.")
            image_padded = image_w_on_b  # Fallback

        # 4. Apply Centering (expects W-on-B input)
        centered_image_pil = image_padded
        if center_digit:
            try:
                logger.debug("📐 Attempting centering (expecting W-on-B)...")
                centered_image_pil = center_digit(image_padded)  # Returns W-on-B
                logger.debug("✅ Applied center_digit")
                # Save intermediate centered PIL (W-on-B)
                try:
                    os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
                    save_path = os.path.join(
                        DEBUG_IMG_DIR, "dbg_canvas_intermediate_centered.png"
                    )
                    centered_image_pil.save(save_path)
                    logger.info(f"🎯 Saved centered PIL: {save_path}")
                except Exception as e:
                    logger.error(f"Err save centered: {e}")
            except Exception as e:
                logger.error(f"⚠️ Centering failed: {e}. Using padded.")
        else:
            logger.warning("⚠️ center_digit unavailable.")

        image_to_process = centered_image_pil  # W-on-B PIL

        # 5. Resize and Convert to Tensor
        resize_totensor = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),  # W(255)->1.0; B(0)->0.0
            ]
        )
        tensor_unnormalized = resize_totensor(image_to_process)
        # Tensor: High=digit, Low=bg

        # 6. Save Unnormalized Debug Image (PNG: White digit, Black bg)
        try:
            os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
            save_path = os.path.join(DEBUG_IMG_DIR, "dbg_canvas_input_unnormalized.png")
            if tensor_unnormalized is not None and tensor_unnormalized.dim() == 3:
                torchvision.utils.save_image(
                    torch.clamp(tensor_unnormalized.clone(), 0, 1), save_path
                )
                logger.info(f"📸 Saved unnormalized tensor: {save_path}")
        except Exception as e:
            logger.error(f"Err save unnorm tensor: {e}")

        # 7. Apply Normalization
        normalize_transform = transforms.Normalize(MNIST_MEAN, MNIST_STD)
        tensor_normalized = normalize_transform(tensor_unnormalized)
        # Tensor: High(digit)->Pos; Low(bg)->Neg :: CORRECT for model

        # 8. Save Normalized Debug Image (PNG appearance varies)
        try:
            os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
            save_path = os.path.join(DEBUG_IMG_DIR, "dbg_canvas_input_normalized.png")
            if tensor_normalized is not None and tensor_normalized.dim() == 3:
                torchvision.utils.save_image(tensor_normalized.clone(), save_path)
                logger.info(f"💾 Saved normalized tensor: {save_path}")
        except Exception as e:
            logger.error(f"Err save norm tensor: {e}")

        # 9. Add batch dimension
        return tensor_normalized.unsqueeze(0)

    except Exception as e:
        logger.error(f"💥 Error in preprocessing: {e}", exc_info=True)
        return None


# --- Flask Routes ---
@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the API is running.

    Returns:
        - JSON response with status and model loaded status.
    """
    status = "healthy" if MODEL_LOADED else "warning_model_not_loaded"
    return jsonify({"status": status, "model_loaded": MODEL_LOADED})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the digit from the provided image.
    Expects a JSON payload with a base64-encoded image.

    Returns:
        - JSON response with prediction, confidence, and timing info.
    """
    predict_start_time = time.time()
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "Missing 'image' key"}), 400

    preprocess_start_time = time.time()
    image_tensor = preprocess_image(data["image"])
    preprocess_time_ms = (time.time() - preprocess_start_time) * 1000

    if image_tensor is None:
        return jsonify({"error": "Preprocessing failed"}), 400

    device = torch.device("cpu")  # Inference on CPU
    image_tensor = image_tensor.to(device)
    model_instance.to(device)

    inference_start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model_instance(image_tensor)
            scaled_logits = outputs / max(TEMPERATURE, 1e-6)  # Apply T
            probabilities = F.softmax(scaled_logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_digit = predicted_class.item()
            confidence_score = confidence.item()
        inference_time_ms = (time.time() - inference_start_time) * 1000
        total_predict_time_ms = (time.time() - predict_start_time) * 1000

        logger.info(
            f"✅ Prediction: {predicted_digit}, "
            f"Calib. Conf: {confidence_score:.4f}, "
            f"Proc: {preprocess_time_ms:.1f}ms, "
            f"Infer: {inference_time_ms:.1f}ms, "
            f"Total: {total_predict_time_ms:.1f}ms "
            f"(T={TEMPERATURE:.3f})"
        )
        return jsonify(
            {
                "prediction": predicted_digit,
                "confidence": confidence_score,
                "preprocessing_time_ms": preprocess_time_ms,
                "inference_time_ms": inference_time_ms,
                "total_time_ms": total_predict_time_ms,
            }
        )
    except Exception as e:
        logger.error(f"💥 Error during inference: {e}", exc_info=True)
        return jsonify({"error": "Inference failed"}), 500


# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting Flask server on port {port}")
    # Use host='0.0.0.0' to run. Gunicorn handles production via Dockerfile.
    app.run(host="0.0.0.0", port=port, debug=False)
