import torch

from pepsflow.models.tensors import Tensors


class Observables:
    """
    Class to compute observables of a PEPS state.
    """

    @staticmethod
    def E(A: torch.Tensor, H: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute the energy of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            H (torch.Tensor): Hamiltonian operator.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm.
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm.
        """
        #           /
        #  A =  -- o --  [D, d, d, d, d]
        #         /|
        #
        #        _|_
        #  H =  |___|  [D², D²]
        #         |
        #
        #  C =  o --  [χ, χ]
        #       |
        #
        #       |
        #  E =  o --  [χ, D², χ]
        #       |
        Rho = Tensors.rho(A, C, T)
        norm = Rho.trace()
        return torch.mm(Rho, H).trace() / norm

    @staticmethod
    def M(A: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the magnetization of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm.
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm.
        """
        Sx, Sy, Sz = (
            Tensors.Mpx().to(A.device),
            Tensors.Mpy().to(A.device),
            Tensors.Mpz().to(A.device),
        )
        Rho = Tensors.rho(A, C, T)
        Tnorm = Rho.trace()
        Mx = torch.mm(Rho, Sx).trace() / Tnorm
        My = torch.mm(Rho, Sy).trace() / Tnorm
        Mz = torch.mm(Rho, Sz).trace() / Tnorm
        return Mx, My, Mz

    @staticmethod
    def xi(T: torch.Tensor) -> torch.Tensor:
        """
        Return the value of the correlation length of the system.

        Args:
            T (torch.Tensor): Edge tensor obtained in CTMRG algorithm.
        """
        M = torch.einsum("abc,dec->adbe", T, T)
        M = M.reshape(T.size(0) ** 2, T.size(0) ** 2)
        w = torch.linalg.eigvalsh(M)
        xi = 1 / torch.log(torch.abs(w[-1]) / torch.abs(w[-2]))
        return xi
