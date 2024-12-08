import numpy as np
from dataclasses import dataclass
import torch
from typing import Sequence
from scipy.stats import ortho_group


@dataclass
class Tensors:

    @staticmethod
    def random_unitary(d: int) -> torch.Tensor:
        """
        Return a random unitary matrix of size d x d.
        """
        return torch.Tensor(ortho_group.rvs(d)).double()

    @staticmethod
    def sx() -> torch.Tensor:
        """
        Return the Pauli X matrix.
        """
        return torch.Tensor([[0, 1], [1, 0]]).double()

    @staticmethod
    def sy() -> torch.Tensor:
        """
        Return the Pauli Y matrix.
        """
        return torch.Tensor([[0, -1], [1, 0]]).double()

    @staticmethod
    def sz() -> torch.Tensor:
        """
        Return the Pauli Z matrix.
        """
        return torch.Tensor([[1, 0], [0, -1]]).double()

    @staticmethod
    def sp() -> torch.Tensor:
        """
        Return the raising operator.
        """
        return torch.Tensor([[0, 1], [0, 0]]).double()

    @staticmethod
    def sm() -> torch.Tensor:
        """
        Return the lowering operator.
        """
        return torch.Tensor([[0, 0], [1, 0]]).double()

    @staticmethod
    def I() -> torch.Tensor:
        """
        Return the identity matrix.
        """
        return torch.eye(2).double()

    @staticmethod
    def Hamiltonian(model: str, **kwargs: dict) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the specified model.

        Args:
            model (str): Name of the model.
            **kwargs (dict): Additional arguments required to construct the Hamiltonian.
        """
        match model:
            case "Ising":
                return Tensors.H_Ising(kwargs["lam"])
            case "Heisenberg":
                return Tensors.H_Heisenberg()
            case _:
                raise ValueError(f"Model {model} not recognized.")

    @staticmethod
    def H_Ising(lam: float) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Ising model
        """
        return -torch.kron(Tensors.sz(), Tensors.sz()) - 0.25 * lam * (
            torch.kron(Tensors.sx(), Tensors.I()) + torch.kron(Tensors.I(), Tensors.sx())
        )

    @staticmethod
    def H_Heisenberg(Jz: float = 1.0, Jxy: float = 1.0) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Heisenberg model.
        """
        sx, sz, sp, sm = Tensors.sx() * 0.5, Tensors.sz() * 0.5, Tensors.sp(), Tensors.sm()

        return 2 * Jz * torch.kron(sz, 4 * sx @ sz @ sx) - Jxy * (
            torch.kron(sm, 4 * sx @ sp @ sx) + torch.kron(sp, 4 * sx @ sm @ sx)
        )

    @staticmethod
    def Mp() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the operator to measure the magnetization in the (x, y, z) direction.
        """
        return (
            torch.kron(Tensors.sx(), torch.eye(2)).double(),
            torch.kron(torch.Tensor([[0, -1], [1, 0]]), torch.eye(2)).double(),
            torch.kron(Tensors.sz(), torch.eye(2)).double(),
        )

    @staticmethod
    def A_random_symmetric(D=2) -> torch.Tensor:
        """
        Return a random rank 5 tensor with legs of size d, which has left-right,
        up-down and diagonal symmetry. The legs are ordered as follows:
        A(phy, up, left, down, right).
        """
        A = torch.rand(size=(2, D, D, D, D), dtype=torch.float64)
        A = Methods.symmetrize_rank5(A)
        return A / A.norm()

    @staticmethod
    def random(shape: tuple) -> torch.Tensor:
        """
        Return a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = torch.rand(size=shape, dtype=torch.float64)
        c = Methods.symmetrize(c)
        return c / c.norm()

    @staticmethod
    def rho(A: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Compute the reduced density matrix of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm.
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm.

        Returns:
            torch.Tensor: Reduced density (ρ) matrix of the PEPS state.
        """

        #           /
        #  A =  -- o --  [D, d, d, d, d]
        #         /|
        #
        #  C =  o --  [χ, χ]
        #       |
        #
        #       |
        #  E =  o --  [χ, D², χ]
        #       |

        D, d = A.size(0), A.size(1)
        a = torch.einsum("mefgh,nabcd->eafbgchdmn", (A, A)).view(d**2, d**2, d**2, d**2, D, D)
        #      /        /              /
        #  -- o --  -- o --   🡺   -- o --  [d², d², d², d², D, D]
        #    /|       /|             /||

        Rho = torch.einsum(
            "ab,bcd,afe,cfghij,dk,khl,emn,no,gmprst,opq,lru,uq->isjt", (C, E, E, a, C, E, E, C, a, E, E, C)
        ).reshape(D**2, D**2)
        #  o -- o -- o
        #  |    |    |
        #  o -- o -- o                        ___
        #  |    |\\  |   [D, D, D, D]   🡺   |___|   [D², D²]
        #  o -- o -- o                       |   |
        #  |    |\\  |
        #  o -- o -- o

        return 0.5 * (Rho + Rho.t())


@dataclass
class Methods:
    """This class contains methods for torch.Tensors, required for the CTM algorithm."""

    @staticmethod
    def symmetrize(M: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize the array about the first two axes. Only works for 2 or 3
        dimensional arrays.
        """
        rank = len(M.shape)

        if rank == 2:
            axes = (1, 0)
        elif rank == 3:
            axes = (2, 1, 0)
        elif rank == 4:
            axes = (3, 1, 2, 0)

        return (M + M.permute(*axes)) / 2

    @staticmethod
    def symmetrize_rank5(A: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize the rank 5 tensor A about all axes, except the physical.
        Legs are ordered as follows: A(phy, up, left, down, right).
        """
        # left-right symmetry
        Asymm = (A + A.permute(0, 1, 4, 3, 2)) / 2.0
        # up-down symmetry
        Asymm = (Asymm + Asymm.permute(0, 3, 2, 1, 4)) / 2.0
        # skew-diagonal symmetry
        Asymm = (Asymm + Asymm.permute(0, 4, 3, 2, 1)) / 2.0
        # diagonal symmetry
        Asymm = (Asymm + Asymm.permute(0, 2, 1, 4, 3)) / 2.0

        return Asymm / Asymm.norm()

    @staticmethod
    def normalize(M: torch.Tensor) -> torch.Tensor:
        """
        Divide all elements in the given array by its largest value.
        """
        return M / M.amax()

    @staticmethod
    def perturb(T: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        """
        Add a small perturbation to the given tensor T.

        Args:
            T (torch.Tensor): Tensor to perturb.
            eps (float): Perturbation strength.
        """
        return T + eps * torch.rand_like(T)

    @staticmethod
    def convert_to_tensor(
        tensors: Sequence[np.ndarray] | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor]:
        """
        If the argument is a numpy array, convert it to a torch tensor.

        Args:
            tensors (Sequence[np.ndarray] | Sequence[torch.Tensor]): List of tensors to
                convert to torch tensors.
        """
        converted_tensors = []
        for tensor in tensors:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            converted_tensors.append(tensor)

        return tuple(converted_tensors)
