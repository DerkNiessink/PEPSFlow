import torch
import time
import matplotlib.pyplot as plt

einsum_times = []
tensordot_times = []

sizes = [size for size in range(100, 500, 20)]

for size in sizes:
    matrix_size = (size, size)

    A = torch.randn(matrix_size)
    B = torch.randn(matrix_size)

    # Measure performance for einsum
    start_einsum = time.time()
    C_einsum = torch.einsum("ij,jk->ik", A, B)
    einsum_time = time.time() - start_einsum

    # Measure performance for tensordot
    start_tensordot = time.time()
    C_tensordot = torch.tensordot(A, B, dims=1)
    tensordot_time = time.time() - start_tensordot

    einsum_times.append(einsum_time)
    tensordot_times.append(tensordot_time)


for einsum_time, tensordot_time, size in zip(einsum_times, tensordot_times, sizes):
    print(f"Matrix size: {size}x{size}")
    print(f"einsum: {einsum_time:.6f} seconds")
    print(f"tensordot: {tensordot_time:.6f} seconds")
    print()

plt.figure(figsize=(10, 5))
plt.plot(sizes, einsum_times, label="einsum Time", marker="o")
plt.plot(sizes, tensordot_times, label="tensordot Time", marker="o")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of einsum vs. tensordot on CPU")
plt.legend()
plt.grid()
plt.show()
