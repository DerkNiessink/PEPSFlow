import numpy as np
from dataclasses import dataclass


@dataclass
class Tensors:

    @staticmethod
    def A_solution() -> np.ndarray:
        """
        Returns the solution state of the Ising model. The legs correspond
        to (down, left, top, right, physical)
        """
        return np.loadtxt("solution_state.txt").reshape(2, 2, 2, 2, 2)

    @staticmethod
    def a_solution() -> np.ndarray:
        """
        Returns contraction of the solution state of the Ising model.
        """
        A = np.loadtxt("solution_state.txt").reshape(2, 2, 2, 2, 2)
        return np.tensordot(A, A, axes=(4, 4)).reshape(4, 4, 4, 4)

    def a(A) -> np.ndarray:
        """
        Returns the contraction of the given rank 5 tensor A.
        """
        return np.tensordot(A, A, axes=(4, 4)).reshape(4, 4, 4, 4)

    @staticmethod
    def H(lam) -> np.ndarray:
        """
        Returns the Hamiltonian operator of the Ising model
        """
        sz = np.array([[1, 0], [0, -1]])
        sx = np.array([[0, 1], [1, 0]])
        I = np.eye(2)

        return -np.kron(sz, sz) - 0.25 * lam * (np.kron(sx, I) + np.kron(I, sx))

    @staticmethod
    def A_random_symmetric(d=2) -> np.ndarray:
        """
        Returns a random rank 5 tensor with legs of size d, which has left-right,
        up-down and diagonal symmetry. The legs are ordered as follows:
        A(phy, up, left, down, right)
        """
        A = np.random.uniform(size=(d, d, d, d, d)) - 0.5
        return Methods.symmetrize_rank5(A)

    @staticmethod
    def random(shape: tuple) -> np.ndarray:
        """
        Returns a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = np.random.uniform(size=shape)
        return Methods.symmetrize(c)


@dataclass
class Methods:
    """This class contains methods for np.arrays, required for the CTM algorithm."""

    @staticmethod
    def symmetrize(M: np.ndarray) -> np.ndarray:
        """
        Symmetrize the array about the first two axes. Only works for 2 or 3
        dimensional arrays.
        """
        rank = len(M.shape)
        if rank != 2 and rank != 3:
            raise Exception("M has to be a 2 or 3 dimensional array.")

        axes = (1, 0) if rank == 2 else (1, 0, 2)
        return np.array(M + np.transpose(M, axes)) / 2

    @staticmethod
    def symmetrize_rank5(A: np.ndarray) -> np.ndarray:
        """
        Symmetrize the rank 5 tensor A about the all axes, except the physical.
        Legs are ordered as follows: A(phy, up, left, down, right).
        """
        Asymm = (A + A.transpose(0, 1, 4, 3, 2)) / 2.0  # left-right symmetry
        Asymm = (Asymm + Asymm.transpose(0, 3, 2, 1, 4)) / 2.0  # up-down symmetry
        Asymm = (Asymm + Asymm.transpose(0, 4, 3, 2, 1)) / 2.0  # skew-diagonal symmetry
        Asymm = (Asymm + Asymm.transpose(0, 2, 1, 4, 3)) / 2.0  # diagonal symmetry
        return Asymm / np.linalg.norm(Asymm)

    @staticmethod
    def normalize(M: np.ndarray) -> np.ndarray:
        """
        Divide all elements in the given array by its largest value.
        """
        return np.array(M / np.amax(M))
