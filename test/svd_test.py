import torch
import time
import matplotlib.pyplot as plt

# Lists to store performance times
gpu_times = []
cpu_times = []

# Define matrix sizes to test
sizes = [size for size in range(100, 2000, 50)]

for size in sizes:
    matrix_size = (size, size)

    # Create a large random matrix
    A = torch.randn(matrix_size)

    # Measure CPU performance for SVD
    start_cpu = time.time()
    U_cpu, S_cpu, V_cpu = torch.linalg.svd(A)
    cpu_time = time.time() - start_cpu

    # Move the matrix to GPU
    A_gpu = A.to("cuda")

    # Measure GPU performance for SVD
    start_gpu = time.time()
    U_gpu, S_gpu, V_gpu = torch.linalg.svd(A_gpu)
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
plt.title("Performance of SVD on CPU vs GPU")
plt.legend()
plt.grid()
plt.show()
