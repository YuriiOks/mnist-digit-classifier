import os, time, torch, torch.nn as nn, torch.optim as optim, numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set recommended environment variables for MPS on Apple Silicon
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Use maximum available memory
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"         # Enable fallback for unsupported ops

def print_header(title):
    length = len(title)
    print("\n" + "=" * (length + 8))
    print(f"🚀  {title}  🚀")
    print("=" * (length + 8) + "\n")

def benchmark_matrix_operations(num_iters=10):
    print_header("Matrix Operations Benchmark (Averaged over {} iterations)".format(num_iters))
    sizes = [1000, 2000, 4000]
    results = []
    speedups = []
    for size in sizes:
        # Create random matrices on CPU
        A_cpu = torch.randn(size, size)
        B_cpu = torch.randn(size, size)
        
        # Time CPU operation over multiple iterations
        cpu_times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = torch.matmul(A_cpu, B_cpu)
            cpu_times.append(time.perf_counter() - start)
        cpu_time = sum(cpu_times) / num_iters
        
        # Skip MPS if not available
        if not torch.backends.mps.is_available():
            results.append([f"{size}x{size}", f"{cpu_time:.4f}s", "N/A", "N/A"])
            continue
        
        device = torch.device("mps")
        A_mps = A_cpu.to(device)
        B_mps = B_cpu.to(device)
        
        # Warmup: run a few iterations to compile kernels
        for _ in range(5):
            _ = torch.matmul(A_mps, B_mps)
        # Time MPS operation over multiple iterations
        mps_times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            _ = torch.matmul(A_mps, B_mps)
            torch.mps.synchronize()
            mps_times.append(time.perf_counter() - start)
        mps_time = sum(mps_times) / num_iters
        speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
        speedups.append(speedup)
        results.append([f"{size}x{size}", f"{cpu_time:.4f}s", f"{mps_time:.4f}s", f"{speedup:.2f}x"])
    
    print(tabulate(results, headers=["Matrix Size", "CPU Time", "MPS Time", "Speedup"], tablefmt="fancy_grid"))
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n🔥 Average Matrix Multiplication Speedup: {avg_speedup:.2f}x")
    return results

def benchmark_cnn_inference(num_iters=10):
    print_header("CNN Inference Benchmark (Averaged over {} iterations)".format(num_iters))
    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.fc1 = nn.Linear(7*7*64, 128)
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
    
    for batch in batch_sizes:
        images_cpu = torch.randn(batch, 1, 28, 28)
        model_cpu = SimpleCNN()
        model_cpu.eval()
        # Warmup and time CPU inference
        cpu_times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_cpu(images_cpu)
            cpu_times.append(time.perf_counter() - start)
        cpu_time = sum(cpu_times) / num_iters
        
        if not torch.backends.mps.is_available():
            results.append([str(batch), f"{cpu_time:.4f}s", "N/A", "N/A"])
            continue
        
        device = torch.device("mps")
        model_mps = SimpleCNN().to(device)
        model_mps.eval()
        images_mps = images_cpu.to(device)
        # Warmup MPS inference
        for _ in range(5):
            with torch.no_grad():
                _ = model_mps(images_mps)
        mps_times = []
        for _ in range(num_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_mps(images_mps)
            torch.mps.synchronize()
            mps_times.append(time.perf_counter() - start)
        mps_time = sum(mps_times) / num_iters
        speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
        speedups.append(speedup)
        results.append([str(batch), f"{cpu_time:.4f}s", f"{mps_time:.4f}s", f"{speedup:.2f}x"])
    
    print(tabulate(results, headers=["Batch Size", "CPU Time", "MPS Time", "Speedup"], tablefmt="fancy_grid"))
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n🔥 Average CNN Inference Speedup: {avg_speedup:.2f}x")
    return results

def benchmark_mnist_training(num_iters=1, batch_size=128):
    print_header("MNIST Training Benchmark (1 Epoch, Averaged over {} iteration(s))".format(num_iters))
    # Define a simple CNN model for MNIST
    class MNISTModel(nn.Module):
        def __init__(self):
            super(MNISTModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(7*7*64, 128)
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

    # Load MNIST (or use synthetic data if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_dataset = torch.utils.data.Subset(train_dataset, range(2000))
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        synthetic_data = torch.randn(2000, 1, 28, 28)
        synthetic_labels = torch.randint(0, 10, (2000,))
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(synthetic_data, synthetic_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def run_epoch(model, optimizer, device):
        model.train()
        criterion = nn.CrossEntropyLoss()
        start = time.perf_counter()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter() - start

    # Run a warm-up epoch on CPU
    model_cpu = MNISTModel()
    optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)
    for _ in range(2):
        _ = run_epoch(model_cpu, optimizer_cpu, torch.device("cpu"))
    cpu_times = [run_epoch(model_cpu, optimizer_cpu, torch.device("cpu")) for _ in range(num_iters)]
    cpu_time = sum(cpu_times) / num_iters

    if not torch.backends.mps.is_available():
        print(tabulate([["CPU", f"{cpu_time:.4f}s", "N/A", "N/A"]],
                         headers=["Device", "Training Time", "Images/sec", "Speedup"],
                         tablefmt="fancy_grid"))
        return None

    device = torch.device("mps")
    model_mps = MNISTModel().to(device)
    optimizer_mps = optim.Adam(model_mps.parameters(), lr=0.001)
    # Instead of trying to compile on MPS (which isn't supported), we simply skip it.
    if device.type == "mps":
        print("Skipping torch.compile on MPS as it is currently unsupported; proceeding with eager mode.")
    else:
        try:
            model_mps = torch.compile(model_mps)
        except Exception as e:
            print("torch.compile unavailable or failed, proceeding without it.")

    # Warm-up on MPS
    for _ in range(2):
        _ = run_epoch(model_mps, optimizer_mps, device)
    mps_times = [run_epoch(model_mps, optimizer_mps, device) for _ in range(num_iters)]
    mps_time = sum(mps_times) / num_iters

    # Calculate throughput (assuming 2000 images in the subset)
    cpu_imgs_per_sec = 2000 / cpu_time
    mps_imgs_per_sec = 2000 / mps_time
    speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
    results = [
        ["CPU", f"{cpu_time:.4f}s", f"{cpu_imgs_per_sec:.1f}", "1.00x"],
        ["MPS", f"{mps_time:.4f}s", f"{mps_imgs_per_sec:.1f}", f"{speedup:.2f}x"]
    ]
    print(tabulate(results, headers=["Device", "Training Time", "Images/sec", "Speedup"], tablefmt="fancy_grid"))
    return speedup



def visualize_speedups(matrix_speedups, inference_speedups, training_speedup=None):
    print_header("Performance Visualization")
    try:
        plt.figure(figsize=(12, 6))
        # Visualize matrix multiplication speedups
        if matrix_speedups:
            plt.subplot(1, 2, 1)
            sizes = [1000, 2000, 4000]
            plt.bar(range(len(sizes)), [float(r[3].replace('x','')) if r[3] != "N/A" else 0 for r in matrix_speedups], color='skyblue')
            plt.xticks(range(len(sizes)), [f"{s}x{s}" for s in sizes])
            plt.ylabel('Speedup (x times)')
            plt.title('Matrix Multiplication Speedup')
            for i, v in enumerate([float(r[3].replace('x','')) if r[3] != "N/A" else 0 for r in matrix_speedups]):
                plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        # Visualize CNN inference speedups
        if inference_speedups:
            plt.subplot(1, 2, 2)
            batch_sizes = [1, 16, 64, 256]
            plt.bar(range(len(batch_sizes)), [float(r[3].replace('x','')) if r[3] != "N/A" else 0 for r in inference_speedups], color='lightgreen')
            plt.xticks(range(len(batch_sizes)), [f"Batch {s}" for s in batch_sizes])
            plt.ylabel('Speedup (x times)')
            plt.title('CNN Inference Speedup')
            for i, v in enumerate([float(r[3].replace('x','')) if r[3] != "N/A" else 0 for r in inference_speedups]):
                plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
        plt.tight_layout()
        plt.savefig('mps_speedup_comparison.png')
        print("\n📊 Speedup chart saved as 'mps_speedup_comparison.png'")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

def main():
    print_header("PyTorch MPS Acceleration Verification (Refined)")
    
    # Display library versions
    print("📚 Library Versions:")
    print(tabulate([
        ["🐍 Python", os.sys.version.split()[0]],
        ["🔥 PyTorch", torch.__version__],
        ["🖼️  TorchVision", getattr(torch.version, 'cuda', 'N/A')],
        ["🔢 NumPy", np.__version__],
        ["📊 Matplotlib", plt.matplotlib.__version__]
    ], tablefmt="fancy_grid"))
    
    mps_available = torch.backends.mps.is_available()
    if not mps_available:
        print("\n⚠️  MPS acceleration is not available. Running benchmarks on CPU only.")
    
    matrix_speedups = benchmark_matrix_operations(num_iters=10)
    inference_speedups = benchmark_cnn_inference(num_iters=10)
    training_speedup = benchmark_mnist_training(num_iters=1, batch_size=128)
    
    if mps_available and (matrix_speedups or inference_speedups):
        visualize_speedups(matrix_speedups, inference_speedups, training_speedup)
    
    print_header("Verification Summary")
    if mps_available:
        print("✅ MPS acceleration is available!")
        # Calculate overall average speedup (where applicable)
        avg_list = []
        if matrix_speedups:
            m_avg = np.mean([float(r[3].replace('x','')) for r in matrix_speedups if r[3] != "N/A"])
            avg_list.append(m_avg)
        if inference_speedups:
            i_avg = np.mean([float(r[3].replace('x','')) for r in inference_speedups if r[3] != "N/A"])
            avg_list.append(i_avg)
        if training_speedup:
            avg_list.append(training_speedup)
        overall_avg = np.mean(avg_list) if avg_list else None
        if overall_avg:
            print(f"🚀 Overall Average Speedup: {overall_avg:.2f}x")
            if overall_avg >= 5:
                print("🔥 Excellent! MPS is providing significant acceleration (>= 5x).")
                print("👉 Consider using batch sizes of 128-256 for training.")
            elif overall_avg >= 3:
                print("✨ Good acceleration (3-5x).")
                print("👉 Consider using batch sizes of 64-128 for training.")
            else:
                print("⚠️  Modest acceleration (< 3x).")
                print("👉 Consider using batch sizes of 32-64 for training.")
        print("\n⚙️  Recommended environment variables remain:")
        print("   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Use maximum available memory")
        print("   export PYTORCH_ENABLE_MPS_FALLBACK=1         # Fallback for unsupported ops")
    else:
        print("❌ MPS acceleration is not available.")
        print("👉 The project will use CPU for computations. Consider optimizing with multi-threading.")

if __name__ == "__main__":
    main()
