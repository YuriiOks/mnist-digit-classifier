# Model Service 🧠

This directory contains the PyTorch-based Convolutional Neural Network (CNN) model for digit classification, along with the Flask API service that exposes it for inference.

## 📄 Contents

- **`model.py`**: Defines the CNN architecture used for digit classification
- **`train.py`**: Script for training the model on the MNIST dataset
- **`app.py`**: Flask application that serves the model via a REST API
- **`utils/`**: Helper functions for model training, evaluation, and inference 
- **`Dockerfile`**: Instructions for containerizing the model service
- **`requirements.txt`**: Python dependencies for the model service

## 🧮 Model Architecture

The MNIST digit classifier uses a Convolutional Neural Network (CNN) with the following architecture:

```
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout(p=0.25)
  (dropout2): Dropout(p=0.5)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

This model achieves over 99% accuracy on the MNIST test set.

## 🚀 Training

The model is trained using the `train.py` script, which:

1. Loads the MNIST dataset via TorchVision
2. Applies data transformations (normalization, augmentation)
3. Trains the model using cross-entropy loss and the Adam optimizer
4. Performs temperature scaling for better calibrated confidence scores
5. Saves the trained model to disk

To train the model manually:

```bash
cd model
python train.py --epochs 15 --batch-size 64 --lr 0.001 --save-path ../saved_models/mnist_cnn.pt
```

## 🌐 API Service

The `app.py` file implements a Flask application that:

1. Loads the pre-trained model
2. Exposes a `/predict` endpoint for digit classification
3. Preprocesses input images to match MNIST format (28x28 grayscale)
4. Returns predicted digit and confidence score in JSON format
5. Provides a `/health` endpoint for service monitoring

API endpoints:

- **POST /predict**: Accepts base64-encoded images and returns digit predictions
- **GET /health**: Returns service status information

## 🐳 Docker Integration

The model service is containerized using Docker and can be built and run with:

```bash
docker build -t mnist-model-service .
docker run -p 5000:5000 mnist-model-service
```

In the full application, this service is managed alongside the web frontend and database using Docker Compose.

## 🔍 Performance Considerations

- The model uses Apple Silicon MPS acceleration when available for faster training and inference
- CUDA is supported for NVIDIA GPUs
- CPU fallback is provided for environments without GPU acceleration

---

## 📚 Related Documentation

- [Main Project README](../README.md)
- [Web README](../web/README.md)
- [Database README](../database/README.md)
