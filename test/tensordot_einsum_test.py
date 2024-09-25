import torch
import time
import matplotlib.pyplot as plt

# Lists to store performance times
einsum_times = []
tensordot_times = []

# Define matrix sizes to test
sizes = [size for size in range(100, 500, 20)]

for size in sizes:
    matrix_size = (size, size)

    # Create two large random matrices
    A = torch.randn(matrix_size)
    B = torch.randn(matrix_size)

    # Measure performance for einsum
    start_einsum = time.time()
    C_einsum = torch.einsum("ij,jk->ik", A, B)  # Matrix multiplication using einsum
    einsum_time = time.time() - start_einsum

    # Measure performance for tensordot
    start_tensordot = time.time()
    C_tensordot = torch.tensordot(A, B, dims=1)  # Matrix multiplication using tensordot
    tensordot_time = time.time() - start_tensordot

    # Store times for analysis
    einsum_times.append(einsum_time)
    tensordot_times.append(tensordot_time)

# Print results
for einsum_time, tensordot_time, size in zip(einsum_times, tensordot_times, sizes):
    print(f"Matrix size: {size}x{size}")
    print(f"einsum: {einsum_time:.6f} seconds")
    print(f"tensordot: {tensordot_time:.6f} seconds")
    print()

# Optional: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(sizes, einsum_times, label="einsum Time", marker="o")
plt.plot(sizes, tensordot_times, label="tensordot Time", marker="o")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of einsum vs. tensordot on CPU")
plt.legend()
plt.grid()
plt.show()
