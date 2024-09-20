import torch
import numpy as np
from ncon import ncon
import scipy

from pepsflow.models.tensors import Tensors, Methods


class Observables:
    """
    Class to compute observables of a PEPS state.
    """

    @staticmethod
    def E(
        A: torch.Tensor, H: torch.Tensor, C: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the energy of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            H (torch.Tensor): Hamiltonian operator (l -> d^2, r -> d^2).
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm (d -> chi, r -> chi).
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm (u -> chi, d -> chi, r -> d).
        """
        Rho = Tensors.rho(A, C, E)
        Tnorm = Rho.trace()
        return torch.mm(Rho, H).trace() / Tnorm

    @staticmethod
    def M(
        A: list[torch.Tensor | np.ndarray],
        C: list[torch.Tensor | np.ndarray],
        E: list[torch.Tensor | np.ndarray],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute the magnetization of a PEPS state.

        Args:
            A (list[torch.Tensor | np.ndarray]): Symmetric A tensors of the PEPS state.
            C (list[torch.Tensor | np.ndarray]): Corner tensors obtained in CTMRG algorithm.
            E (list[torch.Tensor | np.ndarray]): Edge tensors obtained in CTMRG algorithm.
        """
        Sx, Sy, Sz = Tensors.Mpx(), Tensors.Mpy(), Tensors.Mpz()

        Mx, My, Mz = [], [], []
        for a, c, e in zip(A, C, E):
            a, c, e = Methods.convert_to_tensor([a, c, e])
            Rho = Tensors.rho(a, c, e)
            Tnorm = Rho.trace()
            Mx.append(torch.mm(Rho, Sx).trace() / Tnorm)
            My.append(torch.mm(Rho, Sy).trace() / Tnorm)
            Mz.append(torch.mm(Rho, Sz).trace() / Tnorm)
        return Mx, My, Mz

    @staticmethod
    def xi(T: list[torch.Tensor | np.ndarray]) -> list[torch.Tensor]:
        """
        Return the value of the correlation length of the system.

        Args:
            T (torch.Tensor | np.ndarray): Transfer matrix of the system.
        """
        xi = []
        T = list(Methods.convert_to_tensor(T))
        for t in T:
            M = torch.einsum("abc,dec->adbe", t, t)
            M = M.reshape(t.size(0) ** 2, t.size(0) ** 2)
            w = torch.linalg.eigvalsh(M)
            xi.append(1 / torch.log(torch.abs(w[-1]) / torch.abs(w[-2])))
        return xi
