import time
import torch
from torch.nn.functional import normalize

from pepsflow.models.tensors import Methods
from pepsflow.models.svd import CustomSVD
import opt_einsum as oe

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlgSplit:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm. This is a the 'split'
    version.

    Args:
        chi (int): bond dimension of the edge and corner tensors.
        A (torch.Tensor): initial tensor to insert in the CTM algorithm.
        C (torch.Tensor): initial corner tensor for the CTM algorithm.
        T (torch.Tensor): initial edge tensor for the CTM algorithm.
    """

    def __init__(
        self,
        A: torch.Tensor,
        chi: int = 2,
        C_init: torch.Tensor = None,
        T_init: torch.Tensor = None,
    ):

        self.A = A
        self.max_chi = chi
        d = A.size(0)
        self.d = d
        self.chi = self.d if C_init is None else chi
        self.sv_sums = [0]
        self.trunc_errors = []

        # Initialize the corner tensor, shape: (chi, chi)
        if C_init is None:
            self.C = torch.einsum("aabbcdef->cdef", a).view(d**2, d**2)
        else:
            self.C = C_init
        # Initialize the edge tensor,
        if T_init is None:
            self.T = torch.einsum("aabcdefg->bcdefg", a).view(d**2, d, d, d**2)
        else:
            self.T = T_init

    def exe(self, tol=1e-3, count=10, max_steps=10000):
        """
        Execute the CTM algorithm. For each step, two `A` tensors are inserted,
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
            self.C = symm(norm(self.new_C(U, M)))
            self.T = symm(norm(self.new_T(U)))

            # Save sum of singular values
            self.sv_sums.append(torch.sum(s).item())

            tol_counter += 1 if abs(self.sv_sums[-1] - self.sv_sums[-2]) < tol else 0
            if tol_counter == count:
                break

        # Save the computational time and number of iterations
        self.n_iter = len(self.sv_sums)
        self.exe_time = time.time() - start

    def new_C(self, U: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Insert an `a` tensor and evaluate a corner matrix `new_M` by contracting
        the new corner. Renormalize the new corner with the given `U` matrix.

        `U` (torch.Tensor): The renormalization tensor of shape (chi, d, chi).
        `M` (torch.Tensor): The new corner tensor of shape (chi, d, chi, d).

        Returns a tensor of the new corner of shape (chi, chi)
        """
        return torch.einsum("abc,cbde,fed->af", U, M, U)

    def new_M(self) -> torch.Tensor:
        """
        evaluate the `M`, i.e. the new contracted corner with the inserted `a`
        tensor.

        Returns a tensor of the contracted corner of shape (chi x d, chi x d).
        """
        return torch.einsum(
            "ab,acde,bfgh,mfcij,mglkd->eikjlh", self.C, self.T, self.T, self.A, self.A
        )

    def new_T(self, U: torch.Tensor) -> torch.Tensor:
        """
        Insert an `a` tensor and evaluate a new edge tensor. Renormalize
        the edge tensor by contracting with the given truncated `U` tensor.

        `U` (torch.Tensor): The renormalization tensor of shape (chi, d, chi).

        Returns a tensor of the new edge tensor of shape (chi, chi, d).
        """
        return torch.einsum("abc,cde,befg,hfd->ahg", U, self.T, self.a, U)

    def new_U(self, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a
        singular value decomposition (svd) on the given corner tensor `M`. Using
        this factorization `M` can be written as M = U s V*, where the `U` matrix
        is used for renormalization and the `s` matrix contains the singular
        values in descending order.

        `M` (torch.Tensor): The new contracted corner tensor of shape (chi x d, chi x d).

        Returns `s` and the renormalization tensor of shape (chi, d, chi) which is
        obtained by reshaping `U` in a rank-3 tensor and transposing.
        """
        # Reshape M in a matrix
        M = M.contiguous().view(self.chi * self.d, self.chi * self.d)

        k = self.chi
        U, s, Vh = CustomSVD.apply(M)
        self.trunc_errors.append(
            torch.sum(normalize(s, p=1, dim=0)[self.chi :]).float()
        )

        # Let chi grow if the desired chi is not yet reached.
        if self.chi >= self.max_chi:
            self.chi = self.max_chi
            U = U[:, : self.chi]
            s = s[: self.chi]
        else:
            self.chi *= self.d

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return U.view(k, self.d, self.chi).permute(2, 1, 0), norm(s)
