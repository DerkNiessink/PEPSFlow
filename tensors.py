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

    @staticmethod
    def H() -> np.ndarray:
        """
        Returns the Hamiltonian operator of the Ising model
        """
        sigma_z = np.array([[1, 0], [0, -1]])
        sigma_x = np.array([[0, 1], [1, 0]])
        I = np.eye(2)

        return (
            -np.kron(sigma_z, sigma_z)
            - 0.25 * np.kron(sigma_x, I)
            - 0.25 * np.kron(I, sigma_x)
        ).reshape(2, 2, 2, 2)

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
        if len(M.shape) != 2 and len(M.shape) != 3:
            raise Exception("M has to a 2 or 3 dimensional array.")

        axes = (1, 0) if len(M.shape) == 2 else (1, 0, 2)
        return np.array(M + np.transpose(M, axes)) / 2

    @staticmethod
    def normalize(M: np.ndarray) -> np.ndarray:
        """
        Divide all elements in the given array by its largest value.
        """
        return np.array(M / np.amax(M))
