import numpy as np
from dataclasses import dataclass
import torch
from typing import Sequence


@dataclass
class Tensors:

    @staticmethod
    def A_solution() -> torch.Tensor:
        """
        Returns the solution state of the Ising model. The legs correspond
        to (down, left, top, right, physical)
        """
        return torch.from_numpy(
            np.loadtxt("solution_state.txt").reshape(2, 2, 2, 2, 2)
        ).double()

    @staticmethod
    def a_solution() -> torch.Tensor:
        """
        Returns contraction of the solution state of the Ising model.
        """
        A = torch.from_numpy(
            np.loadtxt("solution_state.txt").reshape(2, 2, 2, 2, 2)
        ).double()
        return torch.tensordot(A, A, dims=([4], [4])).reshape(4, 4, 4, 4)

    def a(A: torch.Tensor) -> torch.Tensor:
        """
        Returns the contraction of the given rank 5 tensor A.

        Args:
            A (torch.Tensor): Rank 5 tensor of the PEPS state (phy, up, left, down, right).

        Returns:
            torch.Tensor: Contraction of the rank 5 tensor A. (up, left, down, right)
        """
        d = A.size(1)
        a = (
            (A.view(2, -1).t() @ A.view(2, -1))
            .contiguous()
            .view(d, d, d, d, d, d, d, d)
        )
        a = a.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(d**2, d**2, d**2, d**2)
        return a / a.norm()

    def C_init(a: torch.Tensor) -> torch.Tensor:
        """
        Returns the initial corner tensor for the CTM algorithm.

        Args:
            a (torch.Tensor): Rank 4 tensor of the PEPS state (up, left, down, right).
        """
        return torch.einsum("abcd ->cd", a)

    def T_init(a: torch.Tensor) -> torch.Tensor:
        """
        Returns the initial edge tensor for the CTM algorithm.

        Args:
            a (torch.Tensor): Rank 4 tensor of the PEPS state (up, left, down, right).
        """
        return torch.einsum("abcd -> acd", a)

    @staticmethod
    def H(
        lam: float, sz: torch.Tensor, sx: torch.Tensor, I: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the Hamiltonian operator of the Ising model
        """
        return -torch.kron(sz, sz) - 0.25 * lam * (
            torch.kron(sx, I) + torch.kron(I, sx)
        )

    @staticmethod
    def Mpx() -> torch.Tensor:
        """
        Return the operator to measure the magnetization in the x-direction.
        """
        return torch.kron(torch.Tensor([[0, 1], [1, 0]]), torch.eye(2)).double()

    @staticmethod
    def Mpy() -> torch.Tensor:
        """
        Returns the operator to measure the magnetization in the y-direction.
        """
        return torch.kron(torch.Tensor([[0, -1], [1, 0]]), torch.eye(2)).double()

    @staticmethod
    def Mpz() -> torch.Tensor:
        """
        Returns the operator to measure the magnetization in the z-direction.
        """
        return torch.kron(torch.Tensor([[1, 0], [0, -1]]), torch.eye(2)).double()

    @staticmethod
    def A_random_symmetric(d=2) -> torch.Tensor:
        """
        Returns a random rank 5 tensor with legs of size d, which has left-right,
        up-down and diagonal symmetry. The legs are ordered as follows:
        A(phy, up, left, down, right).
        """
        A = torch.rand(size=(2, d, d, d, d), dtype=torch.float64) - 0.5
        A = Methods.symmetrize_rank5(A)
        return A / torch.norm(A)

    @staticmethod
    def random(shape: tuple) -> torch.Tensor:
        """
        Returns a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = torch.rand(size=shape, dtype=torch.float64)
        c = Methods.symmetrize(c)
        return c / torch.norm(c)

    @staticmethod
    def rho(A: torch.Tensor, C: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Compute the reduced density matrix of a PEPS state.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm (d -> chi, r -> chi).
            E (torch.Tensor): Edge tensor obtained in CTMRG algorithm (u -> chi, d -> chi, r -> d).

        Returns:
            torch.Tensor: Reduced density matrix of the PEPS state.
        """

        # Convert to torch tensors and to the right leg order:
        # A(phy,u,l,d,r), C(d,r), E(u,r,d)
        E = E.permute(1, 2, 0)

        Da = A.size()
        Td = (
            torch.einsum("mefgh,nabcd->eafbgchdmn", (A, A))
            .contiguous()
            .view(Da[1] ** 2, Da[2] ** 2, Da[3] ** 2, Da[4] ** 2, Da[0], Da[0])
        )

        CE = torch.tensordot(C, E, ([1], [0]))  # C(1d)E(dga)->CE(1ga)
        EL = torch.tensordot(
            E, CE, ([2], [0])
        )  # E(2e1)CE(1ga)->EL(2ega)  use E(2e1) == E(1e2)
        EL = torch.tensordot(EL, Td, ([1, 2], [1, 0]))  # EL(2ega)T(gehbmn)->EL(2ahbmn)
        EL = torch.tensordot(
            EL, CE, ([0, 2], [0, 1])
        )  # EL(2ahbmn)CE(2hc)->EL(abmnc), use CE(2hc) == CE(1ga)

        Rho = (
            torch.tensordot(EL, EL, ([0, 1, 4], [0, 1, 4]))
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(Da[0] ** 2, Da[0] ** 2)
        )

        return 0.5 * (Rho + Rho.t())


@dataclass
class Methods:
    """This class contains methods for np.arrays, required for the CTM algorithm."""

    @staticmethod
    def symmetrize(M: torch.Tensor) -> torch.Tensor:
        """
        Symmetrize the array about the first two axes. Only works for 2 or 3
        dimensional arrays.
        """
        rank = len(M.shape)
        if rank != 2 and rank != 3:
            raise Exception("M has to be a 2 or 3 dimensional array.")

        axes = (1, 0) if rank == 2 else (1, 0, 2)
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

        return Asymm / torch.norm(Asymm)

    @staticmethod
    def normalize(M: torch.Tensor) -> torch.Tensor:
        """
        Divide all elements in the given array by its largest value.
        """
        return M / torch.amax(M)

    @staticmethod
    def perturb(T: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        """
        Add a small perturbation to the given tensor T.

        Args:
            T (torch.Tensor): Tensor to perturb.
            eps (float): Perturbation strength.
        """
        return T + eps * torch.randn_like(T)

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
