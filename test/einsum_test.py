import torch
import time
import matplotlib.pyplot as plt

# Lists to store performance times
gpu_times = []
cpu_times = []

# Define matrix sizes to test
sizes = [size for size in range(10, 3000, 50)]

for size in sizes:
    matrix_size = (size, size)

    # Create two large random matrices
    A = torch.randn(matrix_size)
    B = torch.randn(matrix_size)

    # Measure CPU performance for einsum
    start_cpu = time.time()
    C_cpu = torch.einsum("ij,jk->ik", A, B)  # Matrix multiplication using einsum
    cpu_time = time.time() - start_cpu

    # Move the matrices to GPU
    A_gpu = A.to("cuda")
    B_gpu = B.to("cuda")

    # Measure GPU performance for einsum
    start_gpu = time.time()
    C_gpu = torch.einsum(
        "ij,jk->ik", A_gpu, B_gpu
    )  # Matrix multiplication using einsum
    gpu_time = time.time() - start_gpu

    # Store times for analysis
    cpu_times.append(cpu_time)
    gpu_times.append(gpu_time)

for cpu_time, gpu_time, size in zip(cpu_times, gpu_times, sizes):
    print(f"Matrix size: {size}x{size}")
    print(f"CPU: {cpu_time:.6f} seconds")
    print(f"GPU: {gpu_time:.6f} seconds")
    print()

plt.figure(figsize=(10, 5))
plt.plot(sizes, cpu_times, label="CPU Time", marker="o")
plt.plot(sizes, gpu_times, label="GPU Time", marker="o")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of einsum on CPU vs GPU")
plt.legend()
plt.grid()
plt.show()
