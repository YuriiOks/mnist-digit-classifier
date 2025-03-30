#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark script for MNIST model to establish baseline performance metrics.

This script measures:
1. Model accuracy on standard MNIST test set
2. Inference latency for single images and batches
3. Confusion patterns and misclassifications
4. GPU memory usage and utilization
5. Training speed on small batches
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse

# Add the parent directory to sys.path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import MNISTClassifier

# Configure argument parser
parser = argparse.ArgumentParser(description="Benchmark MNIST model performance")
parser.add_argument(
    "--model_path",
    type=str,
    default="model/saved_models/mnist_classifier.pt",
    help="Path to the saved model",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="Batch size for evaluation"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of worker threads for data loading",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="benchmark_results",
    help="Directory to save benchmark results",
)
parser.add_argument(
    "--use_gpu",
    action="store_true",
    help="Use GPU for inference if available",
)
parser.add_argument(
    "--micro_batches",
    type=int,
    default=10,
    help="Number of micro-batches for training benchmark",
)
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Set device
if args.use_gpu and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (Metal Performance Shaders) on Apple Silicon")
elif args.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"Using CPU with {torch.get_num_threads()} threads")

# Define data transforms
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load MNIST test dataset
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


def load_model(model_path):
    """Load the model from the given path."""
    print(f"Loading model from {model_path}")
    model = MNISTClassifier().to(device)

    try:
        # Try to load the model state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model (predictions will be random)")

    model.eval()
    return model


def measure_inference_time(model, loader, num_runs=100):
    """Measure inference time for the model."""
    print("\n=== Measuring Inference Time ===")
    # Get a single sample for individual inference timing
    single_sample = next(iter(loader))[0][0:1].to(device)

    # Warm-up runs
    for _ in range(10):
        _ = model(single_sample)

    # Time single sample inference
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(single_sample)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    single_sample_time = (time.time() - start_time) / num_runs
    print(
        f"Average inference time for single sample: {single_sample_time * 1000:.2f} ms"
    )

    # Time batch inference
    batch = next(iter(loader))[0].to(device)
    batch_size = batch.shape[0]

    # Warm-up run
    _ = model(batch)

    start_time = time.time()
    for _ in range(10):
        _ = model(batch)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    batch_time = (time.time() - start_time) / 10
    samples_per_second = batch_size / batch_time

    print(f"Batch inference time ({batch_size} samples): {batch_time * 1000:.2f} ms")
    print(f"Throughput: {samples_per_second:.2f} samples/second")

    return {
        "single_sample_time_ms": single_sample_time * 1000,
        "batch_inference_time_ms": batch_time * 1000,
        "batch_size": batch_size,
        "throughput_samples_per_second": samples_per_second,
    }


def measure_training_speed(model, loader, num_batches=10):
    """Measure training speed for the model."""
    print("\n=== Measuring Training Speed ===")
    # Set model to training mode
    model.train()

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get some training data
    train_batches = []
    for data, target in loader:
        train_batches.append((data.to(device), target.to(device)))
        if len(train_batches) >= num_batches:
            break

    # Warm-up
    data, target = train_batches[0]
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Measure training time
    start_time = time.time()
    for data, target in train_batches:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    training_time = time.time() - start_time
    samples_per_second = num_batches * args.batch_size / training_time

    print(
        f"Average training time per batch: {training_time * 1000 / num_batches:.2f} ms"
    )
    print(f"Training throughput: {samples_per_second:.2f} samples/second")

    # Set model back to evaluation mode
    model.eval()

    return {
        "training_time_per_batch_ms": training_time * 1000 / num_batches,
        "training_throughput_samples_per_second": samples_per_second,
    }


def evaluate_accuracy(model, loader):
    """Evaluate model accuracy and generate confusion matrix."""
    print("\n=== Evaluating Model Accuracy ===")
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Compute per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Digit {i}: {acc * 100:.2f}%")

    # Find the most confused pairs
    off_diag = np.copy(cm)
    np.fill_diagonal(off_diag, 0)
    confused_pairs = []
    for _ in range(5):  # Get top 5 confused pairs
        max_idx = np.argmax(off_diag)
        i, j = max_idx // 10, max_idx % 10
        confused_pairs.append((i, j, off_diag[i, j]))
        off_diag[i, j] = 0

    print("\nTop confused pairs (true, predicted, count):")
    for true_digit, pred_digit, count in confused_pairs:
        print(f"True: {true_digit}, Predicted: {pred_digit}, Count: {count}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=4))

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": cm.tolist(),
        "confused_pairs": confused_pairs,
    }


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def analyze_errors(model, loader, num_samples=20):
    """Find and display examples of misclassified digits."""
    print("\n=== Analyzing Error Examples ===")
    model.eval()
    misclassified_examples = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            # Find misclassified examples in this batch
            mask = predicted != target
            misclassified_idx = torch.where(mask)[0]

            for idx in misclassified_idx:
                prob = torch.nn.functional.softmax(output[idx], dim=0)
                misclassified_examples.append(
                    {
                        "image": data[idx].cpu(),
                        "true": target[idx].item(),
                        "pred": predicted[idx].item(),
                        "confidence": prob[predicted[idx]].item(),
                        "correct_class_confidence": prob[target[idx]].item(),
                    }
                )

                if len(misclassified_examples) >= num_samples:
                    break

            if len(misclassified_examples) >= num_samples:
                break

    # Plot misclassified examples
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, example in enumerate(misclassified_examples[:20]):
        img = example["image"].squeeze().numpy()

        # Undo normalization for display
        mean, std = 0.1307, 0.3081
        img = img * std + mean

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {example['true']} Pred: {example['pred']}\n"
            f"Conf: {example['confidence']:.2f}"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "misclassified_examples.png"))
    plt.close()

    return misclassified_examples[:20]


def measure_gpu_stats():
    """Measure GPU memory usage and utilization."""
    stats = {}

    if device.type == "cuda":
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB

        print(f"\n=== GPU Memory Usage ===")
        print(f"Allocated: {memory_allocated:.2f} MB")
        print(f"Reserved: {memory_reserved:.2f} MB")

        stats["memory_allocated_mb"] = memory_allocated
        stats["memory_reserved_mb"] = memory_reserved

    elif device.type == "mps":
        # For MPS, we don't have built-in memory stats
        print("\n=== MPS Memory Usage ===")
        print("Memory statistics not available for MPS")

    return stats


def main():
    """Main function to run all benchmarks."""
    # Load the model
    model = load_model(args.model_path)

    # Run all benchmarks
    inference_stats = measure_inference_time(model, test_loader)
    training_stats = measure_training_speed(model, test_loader, args.micro_batches)
    accuracy_stats = evaluate_accuracy(model, test_loader)
    gpu_stats = measure_gpu_stats()

    # Plot confusion matrix
    plot_confusion_matrix(np.array(accuracy_stats["confusion_matrix"]), args.output_dir)

    # Analyze error examples
    error_examples = analyze_errors(model, test_loader)

    # Compile all results
    benchmark_results = {
        "device": device.type,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "inference": inference_stats,
        "training": training_stats,
        "accuracy": accuracy_stats,
        "gpu": gpu_stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results to file
    import json

    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)
