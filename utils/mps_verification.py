# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: utils/mps_verification.py
# Description: PyTorch MPS acceleration verification script
# Created: 2025-03-26
# Updated: 2025-03-30

import os, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset

# Set recommended environment variables for MPS on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use maximum available memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable fallback for unsupported ops


def print_header(title):
    """Prints a formatted header to the console."""
    try:
        # Use terminal size if possible, otherwise default
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 80  # Default width
    title_len = len(title) + 4  # Account for spaces and emojis
    padding = (columns - title_len) // 2
    padding_char = "="
    print("\n" + padding_char * columns)
    print(padding_char * padding + f"🚀 {title} 🚀" + padding_char * padding)
    print(padding_char * columns + "\n")


def check_mps_availability():
    """Checks and prints MPS availability."""
    print_header("MPS Availability Check")
    mps_available = torch.backends.mps.is_available()
    mps_built = (
        torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else False
    )
    cpu_threads = os.cpu_count()
    current_device = "MPS" if mps_available else "CPU"

    print(
        tabulate(
            [
                ["PyTorch Version", torch.__version__],
                ["MPS Available", "✅ Yes" if mps_available else "❌ No"],
                ["MPS Built", "✅ Yes" if mps_built else "❌ No"],
                ["CPU Thread Count", cpu_threads],
                ["Current Device", current_device],
            ],
            headers=["Property", "Value"],
            tablefmt="fancy_grid",
        )
    )

    if mps_available:
        print("\n✨ Great! Your system supports MPS acceleration for PyTorch.")
    else:
        print("\n⚠️ MPS acceleration not detected. Benchmarks will primarily use CPU.")
    return mps_available


def benchmark_matrix_operations(num_iters=10, use_mps=True):
    """Benchmarks matrix multiplication performance."""
    print_header(f"Matrix Operations Benchmark (Avg. {num_iters} iterations)")
    sizes = [1000, 2000, 4000]
    results = []
    speedups = []

    for size in sizes:
        print(f"\n--- Testing Matrix Size: {size}x{size} ---")
        # --- CPU Benchmark ---
        A_cpu = torch.randn(size, size)
        B_cpu = torch.randn(size, size)
        cpu_times = []
        print("  Benchmarking CPU...")
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = torch.matmul(A_cpu, B_cpu)
            cpu_times.append(time.perf_counter() - start)
        cpu_time = sum(cpu_times) / num_iters
        print(f"  CPU Time: {cpu_time:.4f}s")

        # --- MPS Benchmark ---
        mps_time = float("nan")
        speedup = float("nan")
        if use_mps:
            try:
                device = torch.device("mps")
                A_mps = A_cpu.to(device)
                B_mps = B_cpu.to(device)

                print("  MPS Warm-up...")
                for _ in range(5):  # Warm-up
                    _ = torch.matmul(A_mps, B_mps)
                torch.mps.synchronize()
                print("  Benchmarking MPS...")
                mps_times = []
                for _ in range(num_iters):
                    start = time.perf_counter()
                    _ = torch.matmul(A_mps, B_mps)
                    torch.mps.synchronize()  # Ensure completion
                    mps_times.append(time.perf_counter() - start)
                mps_time = sum(mps_times) / num_iters
                speedup = cpu_time / mps_time if mps_time > 0 else float("inf")
                print(f"  MPS Time: {mps_time:.4f}s (Speedup: {speedup:.2f}x)")
                if not np.isnan(speedup):
                    speedups.append(speedup)
            except Exception as e:
                print(f"  MPS Error for size {size}: {e}")
                mps_time = float("nan")
                speedup = float("nan")
        else:
            print("  MPS skipped.")

        results.append(
            [
                f"{size}x{size}",
                f"{cpu_time:.4f}s",
                f"{mps_time:.4f}s" if not np.isnan(mps_time) else "N/A",
                f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A",
            ]
        )

    print("\n--- Matrix Operations Summary ---")
    print(
        tabulate(
            results,
            headers=["Matrix Size", "CPU Time", "MPS Time", "Speedup"],
            tablefmt="fancy_grid",
        )
    )
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n🔥 Average Matrix Multiplication Speedup: {avg_speedup:.2f}x")
    return results


def benchmark_cnn_inference(num_iters=10, use_mps=True):
    """Benchmarks CNN inference performance."""
    print_header(f"CNN Inference Benchmark (Avg. {num_iters} iterations)")

    class SimpleCNN(nn.Module):  # Define model inside function
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.fc1 = nn.Linear(7 * 7 * 64, 128)
            self.fc2 = nn.Linear(128, 10)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    batch_sizes = [1, 16, 64, 256]
    results = []
    speedups = []

    for batch_size in batch_sizes:
        print(f"\n--- Testing Inference Batch Size: {batch_size} ---")
        # --- CPU Benchmark ---
        images_cpu = torch.randn(batch_size, 1, 28, 28)
        model_cpu = SimpleCNN()
        model_cpu.eval()
        print("  Benchmarking CPU...")
        cpu_times = []
        with torch.no_grad():
            for _ in range(num_iters):
                start = time.perf_counter()
                _ = model_cpu(images_cpu)
                cpu_times.append(time.perf_counter() - start)
        cpu_time = sum(cpu_times) / num_iters
        print(f"  CPU Time: {cpu_time:.4f}s")

        # --- MPS Benchmark ---
        mps_time = float("nan")
        speedup = float("nan")
        if use_mps:
            try:
                device = torch.device("mps")
                model_mps = SimpleCNN().to(device)
                model_mps.eval()
                images_mps = images_cpu.to(device)

                print("  MPS Warm-up...")
                with torch.no_grad():
                    for _ in range(5):  # Warm-up
                        _ = model_mps(images_mps)
                torch.mps.synchronize()
                print("  Benchmarking MPS...")
                mps_times = []
                with torch.no_grad():
                    for _ in range(num_iters):
                        start = time.perf_counter()
                        _ = model_mps(images_mps)
                        torch.mps.synchronize()  # Ensure completion
                        mps_times.append(time.perf_counter() - start)
                mps_time = sum(mps_times) / num_iters
                speedup = cpu_time / mps_time if mps_time > 0 else float("inf")
                print(f"  MPS Time: {mps_time:.4f}s (Speedup: {speedup:.2f}x)")
                if not np.isnan(speedup):
                    speedups.append(speedup)
            except Exception as e:
                print(f"  MPS Error for batch size {batch_size}: {e}")
                mps_time = float("nan")
                speedup = float("nan")
        else:
            print("  MPS skipped.")

        results.append(
            [
                str(batch_size),
                f"{cpu_time:.4f}s",
                f"{mps_time:.4f}s" if not np.isnan(mps_time) else "N/A",
                f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A",
            ]
        )

    print("\n--- CNN Inference Summary ---")
    print(
        tabulate(
            results,
            headers=["Batch Size", "CPU Time", "MPS Time", "Speedup"],
            tablefmt="fancy_grid",
        )
    )
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n🔥 Average CNN Inference Speedup: {avg_speedup:.2f}x")
    return results


def benchmark_mnist_training(
    num_iters=1,
    batch_sizes=[16, 32, 64, 128, 256],
    num_images=2000,
    use_mps=True,
):
    """Benchmarks MNIST training performance across different batch sizes."""
    print_header(
        f"MNIST Training Benchmark (1 Epoch on {num_images} images, Avg. {num_iters} iteration(s))"
    )

    class MNISTModel(nn.Module):  # Define model inside function
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(7 * 7 * 64, 128)
            self.fc2 = nn.Linear(128, 10)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.dropout1(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    # Load MNIST data once
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    try:
        print("Loading MNIST dataset...")
        # Ensure data directory exists
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        full_train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        # Ensure num_images doesn't exceed dataset size
        actual_num_images = min(num_images, len(full_train_dataset))
        if actual_num_images < num_images:
            print(
                f"Warning: Requested {num_images} images, but MNIST train set only has {len(full_train_dataset)}. Using {actual_num_images}."
            )
        # Create a consistent subset
        indices = list(range(actual_num_images))
        train_dataset_subset = Subset(full_train_dataset, indices)
        print(f"Dataset loaded ({actual_num_images} images).")
    except Exception as e:
        print(f"Error loading MNIST: {e}. Using synthetic data.")
        actual_num_images = num_images
        synthetic_data = torch.randn(actual_num_images, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (actual_num_images,))
        train_dataset_subset = TensorDataset(synthetic_data, synthetic_labels)

    training_results = []
    best_mps_speedup = -1
    best_batch_size_for_mps = -1

    # Inner function to run one training epoch
    def run_epoch(model, optimizer, device, current_batch_size, dataset):
        # Create DataLoader inside for the specific batch size and dataset
        # Use num_workers=0 for MPS as recommended by PyTorch docs to avoid issues
        num_workers = 0 if device.type == "mps" else 2
        train_loader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type != "mps"),
        )

        model.train()  # Set model to training mode
        criterion = nn.CrossEntropyLoss()
        start = time.perf_counter()
        print(f"  Training on {device} with batch size {current_batch_size}...")
        batch_count = 0
        total_loss = 0
        # Watch for fallback warnings here in the console output!
        for data, target in train_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)  # More efficient zeroing
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Synchronize MPS device to ensure completion before timing
        if device.type == "mps":
            torch.mps.synchronize()
        duration = time.perf_counter() - start
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(
            f"  Finished in {duration:.4f}s ({batch_count} batches, Avg Loss: {avg_loss:.4f})"
        )
        return duration

    # --- Loop over batch sizes ---
    for batch_size in batch_sizes:
        print(f"\n--- Benchmarking Training for Batch Size: {batch_size} ---")

        # --- CPU Benchmark ---
        model_cpu = MNISTModel()  # Re-initialize model for fair comparison
        optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)
        cpu_device = torch.device("cpu")
        print("  CPU Warm-up...")
        _ = run_epoch(
            model_cpu,
            optimizer_cpu,
            cpu_device,
            batch_size,
            train_dataset_subset,
        )
        print("  Benchmarking CPU...")
        cpu_times = [
            run_epoch(
                model_cpu,
                optimizer_cpu,
                cpu_device,
                batch_size,
                train_dataset_subset,
            )
            for _ in range(num_iters)
        ]
        cpu_time = sum(cpu_times) / num_iters
        cpu_imgs_per_sec = (
            actual_num_images / cpu_time if cpu_time > 0 else float("inf")
        )
        print(f"  CPU Avg Time: {cpu_time:.4f}s ({cpu_imgs_per_sec:.1f} Img/s)")

        # --- MPS Benchmark ---
        mps_time = float("nan")
        mps_imgs_per_sec = float("nan")
        speedup = float("nan")

        if use_mps:
            try:
                mps_device = torch.device("mps")
                model_mps = MNISTModel().to(mps_device)  # Re-initialize model
                optimizer_mps = optim.Adam(model_mps.parameters(), lr=0.001)

                print("  MPS Warm-up...")
                _ = run_epoch(
                    model_mps,
                    optimizer_mps,
                    mps_device,
                    batch_size,
                    train_dataset_subset,
                )
                _ = run_epoch(
                    model_mps,
                    optimizer_mps,
                    mps_device,
                    batch_size,
                    train_dataset_subset,
                )
                print("  Benchmarking MPS...")
                mps_times = [
                    run_epoch(
                        model_mps,
                        optimizer_mps,
                        mps_device,
                        batch_size,
                        train_dataset_subset,
                    )
                    for _ in range(num_iters)
                ]
                mps_time = sum(mps_times) / num_iters
                mps_imgs_per_sec = (
                    actual_num_images / mps_time if mps_time > 0 else float("inf")
                )
                speedup = (
                    cpu_time / mps_time
                    if mps_time > 0 and cpu_time > 0
                    else float("inf")
                )
                print(
                    f"  MPS Avg Time: {mps_time:.4f}s ({mps_imgs_per_sec:.1f} Img/s, Speedup: {speedup:.2f}x)"
                )

                if not np.isnan(speedup) and speedup > best_mps_speedup:
                    best_mps_speedup = speedup
                    best_batch_size_for_mps = batch_size

            except Exception as e:
                print(f"  MPS Error for batch size {batch_size}: {e}")
                mps_time = float("nan")
                mps_imgs_per_sec = float("nan")
                speedup = float("nan")
        else:
            print("  MPS skipped.")

        training_results.append(
            [
                batch_size,
                f"{cpu_time:.4f}s",
                f"{cpu_imgs_per_sec:.1f}",
                f"{mps_time:.4f}s" if not np.isnan(mps_time) else "N/A",
                (
                    f"{mps_imgs_per_sec:.1f}"
                    if not np.isnan(mps_imgs_per_sec)
                    else "N/A"
                ),
                f"{speedup:.2f}x" if not np.isnan(speedup) else "N/A",
            ]
        )

    # --- Print Summary Table ---
    print("\n--- Training Benchmark Summary ---")
    print(
        tabulate(
            training_results,
            headers=[
                "Batch Size",
                "CPU Time",
                "CPU Img/s",
                "MPS Time",
                "MPS Img/s",
                "Speedup",
            ],
            tablefmt="fancy_grid",
        )
    )

    if best_batch_size_for_mps != -1:
        print(
            f"\n🚀 Best MPS Training Performance found with Batch Size: {best_batch_size_for_mps} ({best_mps_speedup:.2f}x Speedup)"
        )
    elif use_mps:
        print(
            "\n⚠️ MPS Training showed no speedup compared to CPU across tested batch sizes."
        )
    else:
        print("\nℹ️ MPS was not used for training benchmark.")

    return (
        best_batch_size_for_mps if best_mps_speedup > 1 else None
    )  # Return best BS only if MPS was faster


def visualize_speedups(matrix_results, inference_results, training_results=None):
    """Visualizes speedup results."""
    print_header("Performance Visualization")
    if not plt:
        print("Matplotlib not found. Skipping visualization.")
        return

    try:
        num_plots = 0
        if matrix_results:
            num_plots += 1
        if inference_results:
            num_plots += 1
        # Add training plot later if needed

        if num_plots == 0:
            print("No results to visualize.")
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]  # Make it iterable if only one plot

        plot_idx = 0

        # Plot matrix multiplication speedups
        if matrix_results:
            ax = axes[plot_idx]
            labels = [r[0] for r in matrix_results]
            speedups = [
                float(r[3].replace("x", "")) if r[3] != "N/A" else 0
                for r in matrix_results
            ]
            ax.bar(labels, speedups, color="skyblue")
            ax.set_ylabel("Speedup (x times)")
            ax.set_title("Matrix Multiplication Speedup")
            ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)  # Add 1x line
            for i, v in enumerate(speedups):
                ax.text(
                    i,
                    v + 0.05 * max(speedups, default=1),
                    f"{v:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plot_idx += 1

        # Plot CNN inference speedups
        if inference_results:
            ax = axes[plot_idx]
            labels = [f"Batch {r[0]}" for r in inference_results]
            speedups = [
                float(r[3].replace("x", "")) if r[3] != "N/A" else 0
                for r in inference_results
            ]
            ax.bar(labels, speedups, color="lightgreen")
            ax.set_ylabel("Speedup (x times)")
            ax.set_title("CNN Inference Speedup")
            ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)  # Add 1x line
            for i, v in enumerate(speedups):
                ax.text(
                    i,
                    v + 0.05 * max(speedups, default=1),
                    f"{v:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            plot_idx += 1

        # Add Training speedup plot (Optional - can be added later if needed)

        plt.tight_layout()
        save_path = "mps_speedup_comparison.png"
        plt.savefig(save_path)
        print(f"\n📊 Speedup chart saved as '{save_path}'")
        plt.close(fig)  # Close the figure to free memory

    except Exception as e:
        print(f"❌ Error creating visualization: {e}")


def main():
    """Runs all MPS verification benchmarks."""
    print_header("PyTorch MPS Acceleration Verification (Revised)")

    # Display library versions
    print("📚 Library Versions:")
    print(
        tabulate(
            [
                ["🐍 Python", os.sys.version.split()[0]],
                ["🔥 PyTorch", torch.__version__],
                ["🖼️ TorchVision", torchvision.__version__],
                ["🔢 NumPy", np.__version__],
                [
                    "📊 Matplotlib",
                    (
                        getattr(plt.matplotlib, "__version__", "N/A")
                        if hasattr(plt, "matplotlib")
                        else "N/A"
                    ),
                ],  # Safer access
            ],
            tablefmt="fancy_grid",
        )
    )

    mps_available = check_mps_availability()

    matrix_results = benchmark_matrix_operations(num_iters=10, use_mps=mps_available)
    inference_results = benchmark_cnn_inference(num_iters=10, use_mps=mps_available)
    # Run training benchmark with a range of batch sizes
    best_training_batch_size = benchmark_mnist_training(
        num_iters=1,
        batch_sizes=[16, 32, 64, 128, 256, 512],  # Expanded range
        num_images=4000,  # Slightly larger subset for potentially better signal
        use_mps=mps_available,
    )

    # Visualize results
    visualize_speedups(matrix_results, inference_results)  # Pass the actual results

    # --- Final Summary ---
    print_header("Verification Summary")
    if mps_available:
        print("✅ MPS acceleration is available.")

        # Provide recommendation based ONLY on the training benchmark result
        if best_training_batch_size:
            print(
                f"🚀 MPS Training was faster than CPU. Best performance at batch size: {best_training_batch_size}."
            )
            print(f"👉 Recommended batch size for training: {best_training_batch_size}")
        else:
            print("⚠️ MPS Training was NOT faster than CPU for any tested batch size.")
            print(
                "👉 Consider using CPU for training unless further optimization is done (e.g., different model/optimizer)."
            )

        # Consistently recommend env vars if MPS is available
        print("\n⚙️ Recommended environment variables for MPS:")
        print("   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0")
        print("   export PYTORCH_ENABLE_MPS_FALLBACK=1")
    else:
        print("❌ MPS acceleration is not available.")
        print("👉 The project will use CPU for computations.")


if __name__ == "__main__":
    main()
