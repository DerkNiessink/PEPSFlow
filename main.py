from tensors import Tensors
from CTM_alg import CtmAlg

from ncon import ncon
import numpy as np


def energy(C: np.ndarray, T: np.ndarray, H: np.ndarray, A: np.ndarray) -> float:
    chi = T.shape[0]
    d = int(np.sqrt(T.shape[2]))

    T1 = ncon([C, T, C], [[-1, 1], [1, 2, -2], [2, -3]])
    T1 = T1.reshape(chi, d, d, chi)

    T2 = ncon([T, T], [[-1, 1, -2], [1, -3, -4]])
    T2 = T2.reshape(chi, d, d, chi, d, d)

    T3 = ncon([T1, T2], [[-1, -2, -3, 1], [1, -4, -5, -6, -7, -8]])

    T4 = ncon(
        [T3, T3], [[1, -1, -2, -3, -4, 2, -5, -6], [2, -7, -8, -9, -10, 1, -11, -12]]
    )
    T5 = ncon(
        [H, A, A, A, A],
        [
            [1, 2, 3, 4],
            [-1, -2, -3, 5, 1],
            [-4, 5, -5, -6, 2],
            [-7, -8, -9, 6, 3],
            [-10, 6, -11, -12, 4],
        ],
    )
    return ncon([T4.reshape(2**12), T5.reshape(2**12)], [[1], [1]])


def norm(C: np.ndarray, T: np.ndarray, A: np.ndarray) -> float:
    chi = T.shape[0]
    d = int(np.sqrt(T.shape[2]))

    T1 = ncon([C, T, C], [[-1, 1], [1, 2, -2], [2, -3]])
    T1 = T1.reshape(chi, d, d, chi)

    T2 = ncon([T, T], [[-1, 1, -2], [1, -3, -4]])
    T2 = T2.reshape(chi, d, d, chi, d, d)

    T3 = ncon([T1, T2], [[-1, -2, -3, 1], [1, -4, -5, -6, -7, -8]])

    T4 = ncon(
        [T3, T3], [[1, -1, -2, -3, -4, 2, -5, -6], [2, -7, -8, -9, -10, 1, -11, -12]]
    )

    T5 = ncon(
        [A, A, A, A],
        [
            [-1, -2, -3, 5, 1],
            [-4, 5, -5, -6, 2],
            [-7, -8, -9, 6, 1],
            [-10, 6, -11, -12, 2],
        ],
    )
    return ncon([T4.reshape(2**12), T5.reshape(2**12)], [[1], [1]])


if __name__ == "__main__":

    a = Tensors.a_solution()
    alg = CtmAlg(a, chi=4, d=2)
    alg.exe(count=10, tol=1e-10)
    energy = energy(alg.C, alg.T, Tensors.H(), Tensors.A_solution())
    norm = norm(alg.C, alg.T, Tensors.A_solution())

    print(energy / norm)
