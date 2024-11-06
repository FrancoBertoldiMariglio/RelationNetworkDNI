import torch
import gc
from time import perf_counter
import numpy as np


def benchmark_matmul(size, runs=5, warmup=3):
    """
    Benchmark matrix multiplication with proper warmup and multiple runs
    """
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nBenchmarking {size}x{size} matrix multiplication:")
    times = []
    peak_memory = 0

    # Warmup runs
    print("Warming up...")
    for _ in range(warmup):
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        torch.cuda.synchronize()
        _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        del x, y, _
        torch.cuda.empty_cache()

    # Actual benchmark runs
    for i in range(runs):
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()

        # Create tensors
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')

        # Ensure GPU is ready
        torch.cuda.synchronize()

        # Time the operation
        start = perf_counter()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = perf_counter()

        # Record metrics
        times.append((end - start) * 1000)  # Convert to ms
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()

    # Calculate statistics
    times = np.array(times)
    print(f"Memory used: {peak_memory / 1e9:.3f} GB")
    print(f"Average time: {np.mean(times):.2f} ms")
    print(f"Std dev: {np.std(times):.2f} ms")
    print(f"Min time: {np.min(times):.2f} ms")
    print(f"Max time: {np.max(times):.2f} ms")

    # Calculate FLOPS (floating point operations per second)
    # Matrix multiplication requires 2*n^3 operations
    flops = 2 * size ** 3
    gflops = (flops / (np.mean(times) / 1000)) / 1e9
    print(f"Performance: {gflops:.2f} GFLOPS")

    return {
        'size': size,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'peak_memory': peak_memory,
        'gflops': gflops
    }


def run_benchmarks():
    print("GPU Info:")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    sizes = [1000, 2000, 4000, 8000]
    results = []

    for size in sizes:
        try:
            result = benchmark_matmul(size)
            results.append(result)
        except RuntimeError as e:
            print(f"Error at size {size}: {str(e)}")
            break

    # Print summary
    print("\nSummary:")
    print("Size    | Time (ms) | Memory (GB) | GFLOPS")
    print("-" * 45)
    for r in results:
        print(f"{r['size']:<8} | {r['avg_time']:8.2f} | {r['peak_memory'] / 1e9:10.3f} | {r['gflops']:8.2f}")


if __name__ == "__main__":
    run_benchmarks()