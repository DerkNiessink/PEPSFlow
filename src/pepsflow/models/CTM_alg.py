import time
import torch
import torch.backends.opt_einsum
from torch.nn.functional import normalize
from rich.progress import Progress, Task

from pepsflow.models.tensors import Methods
from pepsflow.models.svd import CustomSVD

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm.

    Args:
        a (torch.Tensor): initial tensor to insert in the CTM algorithm.
        chi (int): bond dimension of the edge and corner tensors. Default is 2.
        C (torch.Tensor): initial corner tensor for the CTM algorithm. Default is None.
        T (torch.Tensor): initial edge tensor for the CTM algorithm. Default is None.
    """

    def __init__(self, a: torch.Tensor, chi: int = 2, C_init: torch.Tensor = None, T_init: torch.Tensor = None):

        self.a = a
        self.max_chi = chi
        self.d = a.size(0)
        self.chi = self.d if C_init is None else chi
        self.sv_sums = [0]
        self.trunc_errors = []

        self.C = torch.einsum("abcd->cd", a).to(a.device) if C_init is None else C_init
        self.T = torch.einsum("abcd->acd", a).to(a.device) if T_init is None else T_init

    def exe(self, max_steps=10000, progress: Progress = None, task: Task = None):
        """
        Execute the CTM algorithm. For each step, an `a` tensor is inserted,
        from which a new edge and corner tensor is evaluated. The new edge
        and corner tensors are normalized and symmetrized every step.

        `max_steps` (int): maximum number of steps before terminating the
        algorithm when convergence has not yet been reached.
        `progress` (rich.progress.Progress): Progress bar to visualize the
        progress of the algorithm. Default is None.
        `task` (rich.progress.Task): Task to update the progress bar with.
        Default is None.
        """
        start = time.time()
        progress.start_task(task) if progress else None

        for _ in range(max_steps):

            M = torch.einsum("ab,cad,bef,gdfh->cgeh", self.C, self.T, self.T, self.a)
            # -> (chi, d, chi, d)

            U, s = self._new_U(M)
            # -> (chi, d, chi), (chi)

            self.C = symm(norm(torch.einsum("abc,cbde,fed->af", U, M, U)))
            # -> (chi, chi)

            self.T = symm(norm(torch.einsum("abc,cde,befg,hfd->ahg", U, self.T, self.a, U)))
            # -> (chi, chi, d)

            # Save the sum of the singular values
            self.sv_sums.append(torch.sum(s).item())
            progress.update(task, advance=1) if progress else None

        # Save the computational time
        self.exe_time = time.time() - start

    def _new_U(self, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        M = M.contiguous().view(self.chi * self.d, self.chi * self.d)

        k = self.chi
        U, s, Vh = CustomSVD.apply(M)
        self.trunc_errors.append(torch.sum(normalize(s, p=1, dim=0)[self.chi :]).float())

        # Let chi grow if the desired chi is not yet reached.
        if self.chi >= self.max_chi:
            self.chi = self.max_chi
            U = U[:, : self.chi]
            s = s[: self.chi]
        else:
            self.chi *= self.d

        # Reshape U back in a three legged tensor and transpose. Normalize the singular values.
        return U.view(k, self.d, self.chi).permute(2, 1, 0), norm(s)
