import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from pepsflow.iPEPS.trainer import iPEPSTrainer

args = {
    "chi": 32,
    "D": 2,
    "data_fn": "data/d=2/lam_3.0.pth",
    "gpu": False,
    "lam": 3.0,
    "max_iter": 5,
    "runs": 1,
    "lr": 1,
    "epochs": 5,
    "use_prev": False,
    "perturbation": 0.0,
}

average_exec_times = []
std_exec_times = []

for n_threads in range(16, 0, -1):

    torch.set_num_threads(n_threads)
    print(f"Number of threads: {torch.get_num_threads()}")

    exec_times = []
    for _ in range(10):
        start = time.time()
        trainer = iPEPSTrainer(args)
        trainer.exe()
        exec_times.append(time.time() - start)

    average_exec_times.append(np.mean(exec_times))
    std_exec_times.append(np.std(exec_times) / np.sqrt(len(exec_times)))

plt.figure(figsize=(6, 4))
plt.errorbar(
    range(1, 17),
    average_exec_times,
    yerr=std_exec_times,
    fmt="o-",
    capsize=3,
    linewidth=0.5,
)
plt.grid(True)
plt.xticks(range(1, 17))
plt.xlabel("Number of threads")
plt.ylabel("Execution time (s)")
plt.savefig("threads.png")
plt.show()
