import numpy as np
import torch
from typing import Sequence
from scipy.stats import ortho_group


class Tensors:
    """
    Class containing tensors needed for iPEPS and computations of observables.

    Args:
        dtype (torch.dtype): Data type of the tensors.
        device (torch.device): Device where the tensors
    """

    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.dev = device

    def random_unitary(self, d: int) -> torch.Tensor:
        """
        Return a random unitary matrix of size d x d.
        """
        return torch.tensor(ortho_group.rvs(d), dtype=self.dtype, device=self.dev)

    def sx(self) -> torch.Tensor:
        """
        Return the Pauli X matrix.
        """
        return torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.dev)

    def sy(self) -> torch.Tensor:
        """
        Return the Pauli Y matrix.
        """
        return torch.tensor([[0, -1], [1, 0]], dtype=self.dtype, device=self.dev)

    def sz(self) -> torch.Tensor:
        """
        Return the Pauli Z matrix.
        """
        return torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.dev)

    def sp(self) -> torch.Tensor:
        """
        Return the raising operator.
        """
        return torch.tensor([[0, 1], [0, 0]], dtype=self.dtype, device=self.dev)

    def sm(self) -> torch.Tensor:
        """
        Return the lowering operator.
        """
        return torch.tensor([[0, 0], [1, 0]], dtype=self.dtype, device=self.dev)

    def I(self) -> torch.Tensor:
        """
        Return the identity matrix.
        """
        return torch.eye(2, dtype=self.dtype, device=self.dev)

    def Hamiltonian(self, model: str, **kwargs: dict) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the specified model.

        Args:
            model (str): Name of the model.
            **kwargs (dict): Additional arguments required to construct the Hamiltonian.
        """
        match model:
            case "Ising":
                return self.H_Ising(kwargs["lam"])
            case "Heisenberg":
                return self.H_Heisenberg()
            case "J1J2":
                return self.H_J1J2()
            case _:
                raise ValueError(f"Model {model} not recognized.")

    def H_Ising(self, lam: float) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Ising model
        """
        sz, sx, I = self.sz(), self.sx(), self.I()
        H = -torch.kron(sz, sz) - 0.25 * lam * (torch.kron(sx, I) + torch.kron(I, sx))
        return H

    def H_Heisenberg(self) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Heisenberg model.
        """
        rot, sz, sp, sm = self.rot_op(), self.sz(), self.sp(), self.sm()
        H = 0.25 * torch.kron(sz, sz) + 0.5 * (torch.kron(sp, sm) + torch.kron(sm, sp))
        H = H.view(2, 2, 2, 2)
        H = torch.einsum("ki,kjcb,ca->ijab", rot, H, rot)
        return 2 * H.reshape(4, 4)

    def H_J1J2(self, J2) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Heisenberg model.
        """
        sx, sz, sp, sm = self.sx(), self.sz(), self.sp(), self.sm()
        H_nn = self.H_Heisenberg()  # Nearest neighbor interaction

        return

    def rot_op(self) -> torch.Tensor:
        """
        Return the rotation operator.
        """
        return torch.tensor([[0, 1], [-1, 0]], dtype=self.dtype, device=self.dev)

    def Mp(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the operator to measure the magnetization in the (x, y, z) direction.
        """
        return (torch.kron(self.sx(), self.I()), torch.kron(self.sy(), self.I()), torch.kron(self.sz(), self.I()))

    def A_random_symmetric(self, D=2) -> torch.Tensor:
        """
        Return a random rank 5 tensor with legs of size d, which has left-right,
        up-down and diagonal symmetry. The legs are ordered as follows:
        A(phy, up, left, down, right).
        """
        A = torch.rand(size=(2, D, D, D, D), dtype=self.dtype, device=self.dev)
        A = Methods.symmetrize_rank5(A)
        return A / A.norm()

    def random(self, shape: tuple) -> torch.Tensor:
        """
        Return a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = torch.rand(size=shape, device=self.dev, dtype=self.dtype)
        c = Methods.symmetrize(c)
        return c / c.norm()

    def rho(self, A: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def get_torch_float(dtype: str) -> torch.dtype:
        """
        Return the default torch float type.
        """
        match dtype:
            case "half":
                return torch.float16
            case "single":
                return torch.float32
            case "double":
                return torch.float64
            case _:
                raise ValueError(f"Data type {dtype} not recognized.")
