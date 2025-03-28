# MNIST Digit Classifier
# Copyright (c) 2025
# File: model/app.py
# Description: Flask API for serving the MNIST classifier model.
# Created: Earlier Date
# Updated: 2025-03-28 (Corrected preprocessing for W-on-B input)

import os
import base64
import io
import json
import sys
import inspect
import logging
import time

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model.model import MNISTClassifier
    logger.info("✅ Successfully imported MNISTClassifier from model.model")
except ImportError as e:
    logger.critical(f"🔥 CRITICAL ERROR importing MNISTClassifier: {e}")
    sys.exit(1)

try:
    from utils.preprocessing import center_digit
    logger.info("✅ Successfully imported center_digit from utils.preprocessing")
except ImportError:
    logger.error("🔥 Failed to import center_digit. Centering disabled.")
    center_digit = None
except Exception as e_import_center:
     logger.error(f"🔥 Error during center_digit import: {e_import_center}")
     center_digit = None

app = Flask(__name__)

try:
    classifier_file_path = inspect.getfile(MNISTClassifier)
    logger.info(f"🧬 MNISTClassifier class loaded from: {classifier_file_path}")
except Exception as e:
    logger.error(f"⚠️ Error inspecting MNISTClassifier: {e}")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'saved_models')
default_model_path = os.path.join(model_dir, 'mnist_classifier.pt')
model_path = os.environ.get('MODEL_PATH', default_model_path)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
DEBUG_IMG_DIR = 'outputs/debug_images'

model_instance = MNISTClassifier()
MODEL_LOADED = False
try:
    logger.info(f"💾 Attempting to load model from: {model_path}")
    model_instance.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu'))
    )
    model_instance.eval()
    MODEL_LOADED = True
    logger.info(f"✅ Loaded model successfully from {model_path}")
except FileNotFoundError:
    logger.error(f"🔥 ERROR: Model file not found at: {model_path}")
    logger.error("🔥 Using untrained model.")
except Exception as e:
    logger.error(f"⚠️ Could not load model state_dict: {e}")
    logger.error("🔥 Using untrained model.")


def preprocess_image(base64_image_data: str) -> torch.Tensor | None:
    """Decodes, pads, centers, preprocesses image, saves debug steps."""
    try:
        # 1. Decode base64 -> PIL Image (expect W-on-B)
        if ',' in base64_image_data:
            _, encoded = base64_image_data.split(",", 1)
        else:
            encoded = base64_image_data
        image_bytes = base64.b64decode(encoded)
        image_w_on_b = Image.open(io.BytesIO(image_bytes)).convert('L')
        fill_color = 0 # Black background for padding

        # 2. Add Padding
        try:
            padding = 30
            image_padded = ImageOps.expand(image_w_on_b, border=padding,
                                           fill=fill_color)
            logger.debug(f"✅ Added {padding}px padding (black).")
        except Exception as pad_err:
            logger.error(f"⚠️ Padding failed: {pad_err}. Using unpadded.")
            image_padded = image_w_on_b

        # 3. Apply Centering (expects W-on-B input)
        centered_image_pil = image_padded # Default if centering fails/disabled
        if center_digit:
            try:
                logger.debug("📐 Attempting centering (expecting W-on-B)...")
                centered_image_pil = center_digit(image_padded)
                logger.debug("✅ Applied center_digit")

                # --- DEBUG SAVE: AFTER Centering ---
                try:
                    os.makedirs(DEBUG_IMG_DIR, exist_ok=True)
                    save_path = os.path.join(
                        DEBUG_IMG_DIR, 'dbg_canvas_intermediate_centered.png'
                    )
                    centered_image_pil.save(save_path)
                    logger.info(f"🎯 Saved centered PIL: {save_path}")
                except Exception as e_save:
                    logger.error(f"Error saving centered PIL: {e_save}")

            except Exception as center_err:
                 logger.error(f"⚠️ Centering failed: {center_err}. "
                              "Using padded image.")
        else:
            logger.warning("⚠️ center_digit function unavailable.")

        image_to_process = centered_image_pil # Should be W-on-B PIL

        # 4. Resize and Convert to Tensor
        resize_totensor = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),  # White(255)->1.0; Black(0)->0.0
        ])
        tensor_unnormalized = resize_totensor(image_to_process)
        # Tensor: High values=digit, Low values=background

        # 5. Save Unnormalized Debug Image
        try:
            save_path_unnorm = os.path.join(DEBUG_IMG_DIR,
                                  'dbg_canvas_input_unnormalized.png')
            if tensor_unnormalized is not None and tensor_unnormalized.dim()==3:
                torchvision.utils.save_image(
                    torch.clamp(tensor_unnormalized.clone(), 0, 1),
                    save_path_unnorm
                )
                logger.info(f"📸 Saved unnormalized tensor: {save_path_unnorm}")
        except Exception as e_save:
            logger.error(f"Error saving unnormalized tensor img: {e_save}")

        # 6. Apply Normalization
        normalize_transform = transforms.Normalize(MNIST_MEAN, MNIST_STD)
        tensor_normalized = normalize_transform(tensor_unnormalized)
        # Tensor: High values (digit) -> Positive; Low (bg) -> Negative

        # 7. Save Normalized Debug Image
        try:
            save_path_norm = os.path.join(DEBUG_IMG_DIR,
                                'dbg_canvas_input_normalized.png')
            if tensor_normalized is not None and tensor_normalized.dim() == 3:
                torchvision.utils.save_image(tensor_normalized.clone(),
                                             save_path_norm)
                logger.info(f"💾 Saved normalized tensor: {save_path_norm}")
        except Exception as e_save:
            logger.error(f"Error saving normalized tensor img: {e_save}")

        # 8. Add batch dimension
        return tensor_normalized.unsqueeze(0)

    except Exception as e:
        logger.error(f"💥 Error in image preprocessing: {e}", exc_info=True)
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint checking API status."""
    status = "healthy" if MODEL_LOADED else "warning_model_not_loaded"
    return jsonify({"status": status, "model_loaded": MODEL_LOADED})


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for digit prediction."""
    predict_start_time = time.time()
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "Missing 'image' key"}), 400

    preprocess_start_time = time.time()
    image_tensor = preprocess_image(data['image'])
    preprocess_time_ms = (time.time() - preprocess_start_time) * 1000

    if image_tensor is None:
        return jsonify({"error": "Preprocessing failed"}), 400

    device = torch.device('cpu') # Use CPU for inference
    image_tensor = image_tensor.to(device)
    model_instance.to(device)

    inference_start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model_instance(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_digit = predicted_class.item()
            confidence_score = confidence.item()
        inference_time_ms = (time.time() - inference_start_time) * 1000
        total_predict_time_ms = (time.time() - predict_start_time) * 1000

        logger.info(f"✅ Prediction: {predicted_digit}, "
                    f"Conf: {confidence_score:.4f}, "
                    f"Proc: {preprocess_time_ms:.1f}ms, "
                    f"Infer: {inference_time_ms:.1f}ms, "
                    f"Total: {total_predict_time_ms:.1f}ms")
        return jsonify({
            "prediction": predicted_digit, "confidence": confidence_score,
            "preprocessing_time_ms": preprocess_time_ms,
            "inference_time_ms": inference_time_ms,
            "total_time_ms": total_predict_time_ms
        })
    except Exception as e:
        logger.error(f"💥 Error during inference: {e}", exc_info=True)
        return jsonify({"error": "Inference failed"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)