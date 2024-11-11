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
            T (torch.Tensor): Edge tensor obtained in CTMRG algorithm.
        """
        #           /
        #  A =  -- o --  [d, D, D, D, D]
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
        #  T =  o --  [χ, D², χ]
        #       |
        Rho = Tensors.rho(A, C, T)
        E = torch.einsum("ab,ab", Rho, H) / Rho.trace()
        #   ___
        #  |___|        ___
        #  _|_|_   /   |___|
        #  |___|
        return E

    @staticmethod
    def M(A: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the magnetization of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm.
            T (torch.Tensor): Edge tensor obtained in CTMRG algorithm.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Magnetization components (Mx, My, Mz).
        """
        #           /
        #  A =  -- o --  [d, D, D, D, D]
        #         /|
        #
        #  C =  o --  [χ, χ]
        #       |
        #
        #       |
        #  T =  o --  [χ, D², χ]
        #       |

        Sx, Sy, Sz = Tensors.Mp()
        Rho = Tensors.rho(A, C, T)

        norm = Rho.trace()
        Mx = torch.mm(Rho, Sx).trace() / norm
        My = torch.mm(Rho, Sy).trace() / norm
        Mz = torch.mm(Rho, Sz).trace() / norm
        #   ___
        #  |___|        ___
        #  _|_|_   /   |___|
        #  |___|

        return Mx, My, Mz

    @staticmethod
    def xi(T: torch.Tensor) -> torch.Tensor:
        """
        Return the value of the correlation length of the system.

        Args:
            T (torch.Tensor): Edge tensor obtained in CTMRG algorithm.
        """
        #      |
        #  T = o --  [χ, D², χ]
        #      |

        chi = T.size(0)
        M = torch.einsum("abc,dbe->adce", T, T).reshape(chi**2, chi**2)
        #   |    |        |
        #   o -- o   🡺   o   [χ², χ²]
        #   |    |        |

        w = torch.linalg.eigvalsh(M)
        return 1 / torch.log(torch.abs(w[-1]) / torch.abs(w[-2]))
