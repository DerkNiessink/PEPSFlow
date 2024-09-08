from models.tensors import Tensors
from models.CTM_alg import CtmAlg
from models.energy import get_energy
from models.map import map_to_A_tensor

import numpy as np
import scipy as sp


def energy(c: np.ndarray):
    A = map_to_A_tensor(c)
    a = Tensors.a(A)
    H = Tensors.H(lam=4)
    alg = CtmAlg(a, chi=2, d=2)
    alg.exe(count=10, tol=1e-10)
    return get_energy(A, H, alg.C, alg.T)


if __name__ == "__main__":
    bound = np.ones(12)

    res = sp.optimize.minimize(
        energy,
        np.random.rand(12) - 0.5,
        method="L-BFGS-B",
        bounds=sp.optimize.Bounds(-bound, bound),
        tol=1e-8,
        options={"maxiter": 1000},
    )
    print(res)
    print(np.unique(Tensors.A_solution()))

"""
a = Tensors.a_solution()
alg = CtmAlg(a, chi=4, d=2)
alg.exe(count=10, tol=1e-10)
energy = get_energy(Tensors.A_solution(), Tensors.H(lam=4), alg.C, alg.T)
print(energy)
"""
