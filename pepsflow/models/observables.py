import torch

from pepsflow.models.tensors import Tensors


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
        A: torch.Tensor,
        C: torch.Tensor,
        E: torch.Tensor,
        Sx: torch.Tensor,
        Sy: torch.Tensor,
        Sz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the magnetization of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm (d -> chi, r -> chi).
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm (u -> chi, d -> chi, r -> d).
            Sx (torch.Tensor): X Pauli operator (d, d).
            Sy (torch.Tensor): Y Pauli operator (d, d).
            Sz (torch.Tensor): Z Pauli operator (d, d).
        """
        Rho = Tensors.rho(A, C, E)
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
            T (torch.Tensor): Tensor to compute the correlation length.
        """
        M = torch.einsum("abc,dec->adbe", T, T)
        M = M.reshape(T.size(0) ** 2, T.size(2) ** 2)
        w = torch.linalg.eigvalsh(M)
        return 1 / torch.log(torch.abs(w[-1] / w[-2]))
