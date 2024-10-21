import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use("science")

from pepsflow.models.CTM_alg_classic import CtmAlg
from pepsflow.models.tensors import Tensors


plt.figure(figsize=(6, 4))
chi_vals = [2, 4, 16]
for chi in chi_vals:

    max_trunc_errors = []
    max_trunc_errors_std = []
    d_vals = range(2, 8)

    for d in d_vals:
        A = Tensors.A_random_symmetric(d=d)
        alg = CtmAlg(a=Tensors.a(A), chi=chi)
        alg.exe(count=50)
        max_trunc_errors.append(np.mean(alg.trunc_errors))
        max_trunc_errors_std.append(np.std(alg.trunc_errors))

    plt.errorbar(
        d_vals,
        max_trunc_errors,
        yerr=max_trunc_errors_std,
        fmt="o",
        capsize=3,
        markersize=2,
        label=f"$\\chi={chi}$",
    )


plt.xlabel(r"$D$", fontsize=15)
plt.ylabel(r"$\epsilon_t$", fontsize=15)
plt.legend()
plt.savefig("tests/truncation_error")
