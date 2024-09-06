import numpy as np
import scipy.sparse.linalg
import scipy.linalg
from ncon import ncon
import time

from models.tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm.

    Args:
        a (np.ndarray): a tensor.
        C (np.ndarray): Initial corner tensor.
        T (np.ndarray): Initial edge tensor
        chi (int): bond dimension of the edge and corner tensors.
    """

    def __init__(
        self,
        a: np.ndarray,
        chi=2,
        d=2,
    ):
        self.a = a
        self.chi = chi
        self.d = d**2
        self.C = Tensors.random((chi, chi))
        self.T = Tensors.random((chi, chi, self.d))
        self.sv_sums = [0]
        self.exe_time = None

    def exe(self, tol=1e-3, count=10, max_steps=10000):
        """
        Execute the CTM algorithm. For each step, an `a` tensor is inserted,
        from which a new edge and corner tensor is evaluated. The new edge
        and corner tensors are normalized and symmetrized every step.

        `tol` (float): convergence criterion.
        `count` (int): Consecutive times the tolerance has to be satified before
        terminating the algorithm.
        `max_steps` (int): maximum number of steps before terminating the
        algorithm when convergence has not yet been reached.
        """
        start = time.time()
        tol_counter = 0
        for _ in range(max_steps):
            # Compute the new contraction `M` of the corner by inserting an `a` tensor.
            M = self.new_M()

            # Use `M` to compute the renormalization tensor
            U, s = self.new_U(M)

            # Normalize and symmetrize the new corner and edge tensors
            self.C = symm(norm(self.new_C(U)))
            self.T = symm(norm(self.new_T(U)))

            # Save sum of singular values
            self.sv_sums.append(np.sum(s))

            tol_counter += 1 if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol else 0

            if tol_counter == count:
                break

        # Save the computational time and number of iterations
        self.n_iter = len(self.sv_sums)
        self.exe_time = time.time() - start

    def new_C(self, U: np.ndarray) -> np.ndarray:
        """
        Insert an `a` tensor and evaluate a corner matrix `new_M` by contracting
        the new corner. Renormalized the new corner with the given `U` matrix.

        `U` (np.ndarray): The renormalization tensor of shape (chi, d, chi).

        Returns an array of the new corner tensor of shape (chi, chi)
        """
        return np.array(
            ncon(
                [U, self.new_M(), U],
                ([-1, 2, 1], [1, 2, 3, 4], [-2, 4, 3]),
            )
        )

    def new_M(self) -> np.ndarray:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns a array of the contracted corner of shape (chi, d, chi, d).
        """
        return np.array(
            ncon(
                [self.C, self.T, self.T, self.a],
                ([1, 2], [-1, 1, 3], [2, -3, 4], [-2, 3, 4, -4]),
            )
        )

    def new_T(self, U: np.ndarray) -> np.ndarray:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (ndarray): The renormalization tensor of shape (chi, d, chi).

        Returns an array of the new edge tensor of shape (chi, chi, d).
        """
        M = ncon([self.T, self.a], ([-1, -2, 1], [1, -3, -4, -5]))
        return np.array(
            ncon(
                [U, M, U],
                ([-1, 3, 1], [1, 2, 3, 4, -3], [-2, 4, 2]),
            )
        )

    def new_U(self, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a
        singular value decomposition (svd) on the given corner tensor `M`. Using
        this factorization `M` can be written as M = U s V*, where the `U` matrix
        is used for renormalization and the `s` matrix contains the singular
        values in descending order.

        `M` (np.array): The new contracted corner tensor of shape (chi, d, chi, d).

        Returns `s` and the renormalization tensor of shape (chi, d, chi) if
        `trunc` = True, else with shape (chi, d, 2*chi), which is obtained by reshaping
        `U` in a rank-3 tensor and transposing.
        """

        # Reshape M in a matrix
        M = np.reshape(M, (self.chi * self.d, self.chi * self.d))
        k = self.chi

        # Get the chi largest singular values and corresponding singular vector matrix.
        # Truncate down to the desired chi value, if not yet reached.
        U, s, _ = scipy.sparse.linalg.svds(M, k=self.chi, which="LM")

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return np.reshape(U, (k, self.d, self.chi)).T, norm(s)
