import torch
from scipy.stats import ortho_group
import opt_einsum as oe


class Tensors:
    """
    Class containing tensors needed for iPEPS and computations of observables.

    Args:
        dtype (str): Data type of the tensors, either 'half', 'single' or 'double'.
        device (str): Device where the tensors, either 'cpu' or 'cuda'.
        chi (int): Bond dimension of the CTM tensors.
        D (int): Bond dimension of the bulk PEPS tensors.
        d (int): Physical dimension of the tensors, default is 2.
    """

    def __init__(self, dtype: str, device: str, chi: int, D: int, d: int = 2):
        dtype_map = {"half": torch.float16, "single": torch.float32, "double": torch.float64}
        device_map = {"cpu": torch.device("cpu"), "cuda": torch.device("cuda")}
        self.dtype = dtype_map[dtype]
        self.dev = device_map[device]
        expr = "ab,bcde,fgha,kgijc,nhlmd,rjopq,vmstu,equw,ptxy,wy,zA,zBCD,AEFf,HECGi,JFDIl,MGKLo,PINOs,BQKN,LORx,QR->knrvHJMP"
        self.rho_expr = oe.contract_expression(
            expr,
            (chi, chi),
            (chi, D, D, chi),
            (chi, D, D, chi),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (chi, D, D, chi),
            (D, D, chi, chi),
            (chi, chi),
            (chi, chi),
            (chi, chi, D, D),
            (chi, D, D, chi),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (d, D, D, D, D),
            (chi, chi, D, D),
            (D, D, chi, chi),
            (chi, chi),
            optimize="auto-hq",
            memory_limit=4 * 1024**3,  # 4 GB memory limit
        )

    def gauges(self, D: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get gauge transformation matrices.

        Args:
            which (str): Type of gauge transformation. Can be 'unitary', 'invertible' or None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Gauge transformation matrices U1 and U2.
        """
        match which:
            case "unitary":
                g1, g2 = self.random_unitary(D), self.random_unitary(D)
            case "invertible":
                g1, g2 = self.random_tensor((D, D)), self.random_tensor((D, D))
            case None:
                g1 = g2 = self.identity(D)
            case _:
                raise ValueError(f"Unknown gauge type: {which}")
        return g1, g2

    def gauge_transform(self, A: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        Apply gauge transformation to the tensor A.

        Args:
            A (torch.Tensor): Tensor to be transformed.
            g1 (torch.Tensor): First gauge transformation matrix.
            g2 (torch.Tensor): Second gauge transformation matrix.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        g1_inv = torch.linalg.inv(g1)
        g2_inv = torch.linalg.inv(g2)
        return torch.einsum("purdl,Uu,Rr,dD,lL->pURDL", A, g2, g1, g2_inv, g1_inv)

    def A_random(self, D: int) -> torch.Tensor:
        """Return a random state of size [d, D, D, D, D]."""
        return torch.randn(2, D, D, D, D, dtype=self.dtype, device=self.dev)

    def random_tensor(self, shape: tuple) -> torch.Tensor:
        """Return a random tensor of specific shape."""
        return torch.randn(shape, dtype=self.dtype, device=self.dev)

    def random_unitary(self, d: int) -> torch.Tensor:
        """Return a random unitary matrix of size d x d."""
        return torch.tensor(ortho_group.rvs(d), dtype=self.dtype, device=self.dev)

    def identity(self, D: int) -> torch.Tensor:
        """Return an identity matrix of size D x D."""
        return torch.eye(D, dtype=self.dtype, device=self.dev)

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
        Return a random rank 5 tensor with legs of size D, which has left-right,
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
                return self.H_Heis_rot()
            case "J1J2":
                return self.H_Heis_rot()
            case _:
                raise ValueError(f"Model {model} not recognized.")

    def H_Ising(self, lam: float) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Ising model
        """
        sz, sx, I = self.sz(), self.sx(), self.I()
        H = -torch.kron(sz, sz) - 0.25 * lam * (torch.kron(sx, I) + torch.kron(I, sx))
        return H

    def H_Heis_rot(self) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Heisenberg model in rotated basis.
        """
        rot, sz, sp, sm = self.rot_op(), self.sz(), self.sp(), self.sm()
        H = 0.25 * torch.kron(sz, sz) + 0.5 * (torch.kron(sp, sm) + torch.kron(sm, sp))
        H = H.view(2, 2, 2, 2)
        H = torch.einsum("ki,kjcb,ca->ijab", rot, H, rot)
        return 2 * H.reshape(4, 4)

    def H_Heis(self) -> torch.Tensor:
        """
        Return the Hamiltonian operator of the Heisenberg model.
        """
        sz, sp, sm = self.sz(), self.sp(), self.sm()
        H = 0.25 * torch.kron(sz, sz) + 0.5 * (torch.kron(sp, sm) + torch.kron(sm, sp))
        return 2 * H.reshape(4, 4)

    def double(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the double layer of tensor A.

        Args:
            A (torch.Tensor): Bulk A tensor of the PEPS state.

        Returns:
            torch.Tensor: Double layer of the tensor A, which is a rank 6 tensor.
        """
        d, D = A.size(0), A.size(1)
        #      /        /              /
        #  -- o --  -- o --   🡺   -- o --  [D², D², D², D², d, d]
        #    /|       /|             /||
        return torch.einsum("mefgh,nabcd->eafbgchdmn", (A, A)).reshape(D**2, D**2, D**2, D**2, d, d)

    def rho_symmetric(self, A: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute the reduced density matrix of a PEPS state with symmetric tensors.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state.
            C (torch.Tensor): Corner tensor obtained in CTMRG algorithm.
            T (torch.Tensor): Edge tensor obtained in CTMRG algorithm.

        Returns:
            torch.Tensor: Reduced density matrix of the PEPS state.
        """
        D, chi = A.size(1), C.size(0)
        #   o -- o -- o -- o
        #   |   ||    ||   |
        #   o== oo == oo ==o        ||_||
        #   |   ||\\  ||\\ |    🡺  |___|   [d, d, d, d, d, d, d, d]
        #   o== oo == oo ==o        || ||
        #   |   ||\\  ||\\ |
        #   o -- o -- o -- o
        return self.rho_expr(
            C,
            T.view(chi, D, D, chi),
            T.view(chi, D, D, chi),
            A,
            A,
            A,
            A,
            T.view(chi, D, D, chi),
            T.view(chi, D, D, chi).permute(1, 2, 0, 3),
            C,
            C,
            T.view(chi, D, D, chi).permute(0, 3, 1, 2),
            T.view(chi, D, D, chi),
            A,
            A,
            A,
            A,
            T.view(chi, D, D, chi).permute(0, 3, 1, 2),
            T.view(chi, D, D, chi).permute(1, 2, 0, 3),
            C,
        )

    def rho_general(
        self,
        A: torch.Tensor,
        C1: torch.Tensor,
        C2: torch.Tensor,
        C3: torch.Tensor,
        C4: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T3: torch.Tensor,
        T4: torch.Tensor,
    ):
        """
        Compute the reduced density matrix of a PEPS state with general tensors.

        Args:
            A (torch.Tensor): Symmetric A tensor of the PEPS state of shape [d, D, D, D, D].
            C1, C2, C3, C4 (torch.Tensor): Corner tensors obtained in CTMRG algorithm of shape [χ, χ].
            T1, T2, T3, T4 (torch.Tensor): Edge tensors obtained in CTMRG algorithm of shapes [χ, D^2, χ], [χ, χ, D^2],
            [D^2, χ, χ], [χ, D^2, χ] respectively.

        Returns:
            torch.Tensor: Reduced density matrix of the PEPS state.
        """
        D, chi = A.size(1), C1.size(0)
        return self.rho_expr(
            C1,
            T4.view(chi, D, D, chi),
            T1.view(chi, D, D, chi),
            A,
            A,
            A,
            A,
            T4.view(chi, D, D, chi),
            T3.view(D, D, chi, chi),
            C4,
            C2,
            T2.view(chi, chi, D, D),
            T1.view(chi, D, D, chi),
            A,
            A,
            A,
            A,
            T2.view(chi, chi, D, D),
            T3.view(D, D, chi, chi),
            C3,
        )

    def E(self, rho: torch.Tensor, H: torch.Tensor, which: str) -> torch.Tensor:
        """
        Compute the energy of a PEPS state with general interactions.
        Args:
            rho (torch.Tensor): Reduced density matrix of the PEPS state.
            which (str): Type of energy to compute. Can be 'horizontal', 'vertical', 'diagonal' or 'antidiagonal'.
        """
        exprs = {
            "horizontal": "abccdeff->adbe",
            "vertical": "abcdeeff->acbd",
            "diagonal": "abccddef->aebf",
            "antidiagonal": "aabcdeff->bdce",
        }
        if which not in exprs:
            raise ValueError(f"Unknown energy type: {which}")
        d2 = H.size(0)
        rho = torch.einsum(exprs[which], rho).reshape(d2, d2)
        rho = Methods.symmetrize(rho)
        E = torch.einsum("ab,ab", rho, H) / rho.trace()
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

        Rho = self.rho_nn(A, C, T)
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
