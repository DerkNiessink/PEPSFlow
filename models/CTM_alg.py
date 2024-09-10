import time
import torch

from models.tensors import Tensors, Methods

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm.

    Args:
        a (torch.Tensor): a tensor.
        C (torch.Tensor): Initial corner tensor.
        T (torch.Tensor): Initial edge tensor
        chi (int): bond dimension of the edge and corner tensors.
    """

    def __init__(
        self,
        a: torch.Tensor,
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
            self.sv_sums.append(torch.sum(s).item())

            tol_counter += 1 if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol else 0

            if tol_counter == count:
                break

        # Save the computational time and number of iterations
        self.n_iter = len(self.sv_sums)
        self.exe_time = time.time() - start

    def new_C(self, U: torch.Tensor) -> torch.Tensor:
        """
        Insert an `a` tensor and evaluate a corner matrix `new_M` by contracting
        the new corner. Renormalize the new corner with the given `U` matrix.

        `U` (torch.Tensor): The renormalization tensor of shape (chi, d, chi).

        Returns a tensor of the new corner of shape (chi, chi)
        """
        return torch.einsum("abc,cbde,fed->af", U, self.new_M(), U)

    def new_M(self) -> torch.Tensor:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns a tensor of the contracted corner of shape (chi, d, chi, d).
        """
        return torch.einsum("ab,cad,bef,gdfh->cgeh", self.C, self.T, self.T, self.a)

    def new_T(self, U: torch.Tensor) -> torch.Tensor:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (torch.Tensor): The renormalization tensor of shape (chi, d, chi).

        Returns a tensor of the new edge tensor of shape (chi, chi, d).
        """
        M = torch.einsum("abc,cdef->abdef", self.T, self.a)
        return torch.einsum("abc,cdbef,ged->agf", U, M, U)

    def new_U(self, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a
        singular value decomposition (svd) on the given corner tensor `M`. Using
        this factorization `M` can be written as M = U s V*, where the `U` matrix
        is used for renormalization and the `s` matrix contains the singular
        values in descending order.

        `M` (torch.Tensor): The new contracted corner tensor of shape (chi, d, chi, d).

        Returns `s` and the renormalization tensor of shape (chi, d, chi) which is
        obtained by reshaping `U` in a rank-3 tensor and transposing.
        """

        # Reshape M in a matrix
        M = M.reshape(self.chi * self.d, self.chi * self.d)
        k = self.chi

        # Perform SVD and truncate to the desired chi values
        U, s, Vh = torch.svd(M, some=True)
        U = U[:, :k]

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return U.view(k, self.d, self.chi).permute(2, 1, 0), norm(s[:k])
