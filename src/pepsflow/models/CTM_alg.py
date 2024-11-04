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
        A (torch.Tensor): initial tensor to insert in the CTM algorithm. (phy, up, left, down, right) -> (D, d, d, d, d).
        chi (int): bond dimension of the edge and corner tensors. Default is 2.
        C (torch.Tensor): initial corner tensor for the CTM algorithm. Default is None. (right, down) -> (chi, chi).
        T (torch.Tensor): initial edge tensor for the CTM algorithm. Default is None. (up, right, down) -> (chi, d, chi).

                 /                      |
        A =  -- o --   C =  o --   T =  o --
               /|           |           |

    """

    def __init__(
        self, A: torch.Tensor, chi: int, C_init: torch.Tensor = None, T_init: torch.Tensor = None, split: bool = False
    ):
        d = A.size(1)
        self.d = d
        self.split = split
        self.max_chi = chi
        self.chi = d**2 if C_init is None else chi
        self.trunc_errors = []

        a = torch.einsum("abcde,afghi->bfcidheg", A, A)
        # -> (d, d, d, d, d, d, d, d)
        self.C = torch.einsum("aabbcdef->cdef", a).view(d**2, d**2) if C_init is None else C_init
        # -> (chi, chi)

        if T_init is not None:
            self.T = T_init.view(chi, d, d, chi) if split else T_init.view(chi, d**2, chi)
        else:
            shape = (d**2, d, d, d**2) if split else (d**2, d**2, d**2)
            self.T = torch.einsum("aabcdefg->bcdefg", a).view(shape)
        # -> (chi, d, d, chi) if split else (chi, d^2, chi)

        self.a = A if split else a.reshape(d**2, d**2, d**2, d**2)
        # -> (D, d, d, d, d) if split else (d^2, d^2, d^2, d^2)

    def exe(self, N: int = 1, progress: Progress = None, task: Task = None):
        """
        Execute the CTM algorithm for N steps.

        `N` (int): number of steps to execute the algorithm for.
        `progress` (rich.progress.Progress): Progress bar to visualize the progress of the algorithm. Default is None.
        `task` (rich.progress.Task): Task to update the progress bar with. Default is None.
        """
        start = time.time()
        progress.start_task(task) if progress else None
        for _ in range(N):
            self.split_step() if self.split else self.classic_step()
            progress.update(task, advance=1) if progress else None
        self.exe_time = time.time() - start
        if self.split:
            self.T = self.T.view(self.chi, self.d**2, self.chi)

    def classic_step(self):
        """
        Execute one "classic" CTM step. This is the standard CTM algorithm for the rank-4 input tensor.
        """
        # fmt: off
        M = torch.einsum("ab,acd,bef,ecgh->dgfh", self.C, self.T, self.T, self.a)         # -> (chi, d^2, chi, d^2)
        U = self._new_U(M)                                                                # -> (chi, d^2, chi)
        self.C = symm(norm(torch.einsum("abc,abfe,fed->cd", U, M, U)))                    # -> (chi, chi)
        self.T = symm(norm(torch.einsum("cba,cgf,bdeg,feh->adh", U, self.T, self.a, U)))  # -> (chi, d^2, chi)

    def split_step(self):
        """
        Execute one "split" CTM step. This is the CTM algorithm for the rank-5 input tensor.
        """
        # fmt: off
        M = torch.einsum("ab,acde,bfgh,mfcij,mglkd->eikhjl", self.C, self.T, self.T, self.a, self.a)        # -> (chi, d, d, chi, d, d)
        U = self._new_U(M)                                                                                  # -> (chi, d, d, chi)
        self.C = symm(norm(torch.einsum("abcd,abcefh,efhg->dg", U, M, U)))                                  # -> (chi, chi)
        self.T = symm(norm(torch.einsum("abcd,aefg,lbemh,lcfij,gmik->dhjk", U, self.T, self.a, self.a, U))) # -> (chi, d, d, chi)

    def _new_U(self, M: torch.Tensor) -> torch.Tensor:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a singular value decomposition
        (svd) on the given corner tensor `M`. Using this factorization `M` can be written as M = U s V*, where
        the `U` matrix is used for renormalization.

        `M` (torch.Tensor): The new contracted corner tensor of shape (chi, d, chi, d).

        Returns the renormalization tensor of shape (chi, d, chi) which is obtained by reshaping `U` in a rank-3.
        """
        M = M.contiguous().view(self.chi * self.d**2, self.chi * self.d**2)

        k = self.chi
        U, s, _ = CustomSVD.apply(M)
        self.trunc_errors.append(torch.sum(normalize(s, p=1, dim=0)[self.chi :]).float())

        # Let chi grow if the desired chi is not yet reached.
        if self.chi >= self.max_chi:
            self.chi = self.max_chi
            U = U[:, : self.chi]
        else:
            self.chi *= self.d**2

        # Reshape U back in a rank-3 or 4 tensor.
        shape = (k, self.d, self.d, self.chi) if self.split else (k, self.d**2, self.chi)
        return U.view(shape)
