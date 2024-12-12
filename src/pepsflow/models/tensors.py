import torch
from scipy.stats import ortho_group


class Tensors:
    """
    Class containing tensors needed for iPEPS and computations of observables.

    Args:
        dtype (str): Data type of the tensors, either 'half', 'single' or 'double'.
        device (str): Device where the tensors, either 'cpu' or 'cuda'.
    """

    def __init__(self, dtype: str, device: str):
        dtype_map = {"half": torch.float16, "single": torch.float32, "double": torch.float64}
        device_map = {"cpu": torch.device("cpu"), "cuda": torch.device("cuda")}
        self.dtype = dtype_map[dtype]
        self.dev = device_map[device]

    def random_unitary(self, d: int) -> torch.Tensor:
        """Return a random unitary matrix of size d x d."""
        return torch.tensor(ortho_group.rvs(d), dtype=self.dtype, device=self.dev)

    def sx(self) -> torch.Tensor:
        """Return the Pauli X matrix."""
        return torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.dev)

    def sy(self) -> torch.Tensor:
        """Return the Pauli Y matrix."""
        return torch.tensor([[0, -1], [1, 0]], dtype=self.dtype, device=self.dev)

    def sz(self) -> torch.Tensor:
        """Return the Pauli Z matrix."""
        return torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.dev)

    def sp(self) -> torch.Tensor:
        """Return the raising operator."""
        return torch.tensor([[0, 1], [0, 0]], dtype=self.dtype, device=self.dev)

    def sm(self) -> torch.Tensor:
        """Return the lowering operator."""
        return torch.tensor([[0, 0], [1, 0]], dtype=self.dtype, device=self.dev)

    def I(self) -> torch.Tensor:
        """Return the identity matrix."""
        return torch.eye(2, dtype=self.dtype, device=self.dev)

    def rot_op(self) -> torch.Tensor:
        """Return the rotation operator."""
        return torch.tensor([[0, 1], [-1, 0]], dtype=self.dtype, device=self.dev)

    def random(self, shape: tuple) -> torch.Tensor:
        """
        Return a random tensor of specific shape, which can be either rank 2
        or rank 3. The tensor is symmetric under the exchange of the first two
        indices and the values are normalized.
        """
        c = torch.rand(size=shape, device=self.dev, dtype=self.dtype)
        c = Methods.symmetrize(c)
        return c / c.norm()

    def A_random_symmetric(self, D=2) -> torch.Tensor:
        """
        Return a random rank 5 tensor with legs of size d, which has left-right,
        up-down and diagonal symmetry. The legs are ordered as follows:
        A(phy, up, left, down, right).
        """
        A = torch.rand(size=(2, D, D, D, D), dtype=self.dtype, device=self.dev)
        A = Methods.symmetrize_rank5(A)
        return A / A.norm()

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
        Return the Hamiltonian operator of the J1-J2 model.
        """
        sx, sz, sp, sm = self.sx(), self.sz(), self.sp(), self.sm()
        H_nn = self.H_Heisenberg()  # Nearest neighbor interaction

        return

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

    def E(self, A: torch.Tensor, H: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
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
        Rho = self.rho(A, C, T)
        E = torch.einsum("ab,ab", Rho, H) / Rho.trace()
        #   ___
        #  |___|        ___
        #  _|_|_   /   |___|
        #  |___|
        return E

    def M(self, A: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        Rho = self.rho(A, C, T)
        M = lambda pauli_op: torch.mm(Rho, torch.kron(self.I(), pauli_op)).trace() / Rho.trace()
        #   ___
        #  |___|        ___
        #  _|_|_   /   |___|
        #  |___|

        return M(self.sx()), M(self.sy()), M(self.sz())

    def xi(self, T: torch.Tensor) -> torch.Tensor:
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
