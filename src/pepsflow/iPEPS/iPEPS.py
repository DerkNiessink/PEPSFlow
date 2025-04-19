import torch
import numpy as np

from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
from pepsflow.models.tensors import Tensors


from abc import ABC, abstractmethod
import torch
import numpy as np
from pepsflow.models.tensors import Tensors


def make_ipeps(args: dict, initial_ipeps: "iPEPS" = None) -> "iPEPS":
    """
    iPEPS is a class of tensor networks used to represent a quantum many-body state and perform
    variational optimization.

    Args:
        args (dict): Dictionary containing the iPEPS parameters.
        initial_ipeps (iPEPS, optional): Initial iPEPS object to use as a starting point.

    Returns:
        iPEPS: An instance of the iPEPS class with the specified symmetry.
    """
    if args["rotational_symmetry"] in ["both", "ctm"]:
        return RotationalSymmetricIPEPS(args, initial_ipeps)
    elif args["rotational_symmetry"] in ["state", None]:
        return GeneralIPEPS(args, initial_ipeps)
    else:
        raise ValueError(f"Unknown symmetry: {args['rotational_symmetry']}")


class iPEPS(torch.nn.Module, ABC):
    def __init__(self, args: dict, initial_ipeps: "iPEPS" = None):
        super().__init__()
        self.args = args
        self.initial_ipeps = initial_ipeps
        self.tensors = Tensors(args["dtype"], args["device"])
        self.data = {"losses": [], "norms": [], "Niter_warmup": []}
        self.to(args["device"])
        if args["seed"] is not None:
            torch.manual_seed(args["seed"])
            np.random.seed(args["seed"])

        self.H = self.tensors.Hamiltonian(args["model"], lam=args["lam"])
        self.params, self.map = None, None

    def add_data(self, E: torch.Tensor = None, Niter_warmup: int = None):
        """
        Add data of the energy, norm and number of warmup iterations to the iPEPS.

        Args:
            E (torch.Tensor): Energy of the iPEPS tensor network.
            Niter_warmup (int): Number of warmup iterations.
        """
        self.data["losses"].append(E)
        # squared_norm = sum(p.data.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
        # self.data["norms"].append(torch.sqrt(squared_norm) if isinstance(squared_norm, torch.Tensor) else squared_norm)

    def do_warmup_steps(self) -> tuple[torch.Tensor, ...]:
        """Warmup the iPEPS tensor by performing the CTM algorithm without gradient tracking.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["warmup_steps"], grad=False)

    def do_gradient_steps(self, tensors) -> tuple[torch.Tensor, ...]:
        """Take gradient steps in the optimization of the iPEPS tensor.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["Niter"], grad=True, tensors=tensors)

    def do_evaluation(self) -> tuple[torch.Tensor, ...]:
        """ "Evaluate the iPEPS tensor by performing the CTM algorithm without gradient tracking.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["Niter"], grad=False)

    @abstractmethod
    def _setup_random(self):
        """Setup the iPEPS tensor with a random initial state."""
        pass

    @abstractmethod
    def _setup_from_initial_ipeps(self):
        """Setup the iPEPS tensor from an initial state."""
        pass

    @abstractmethod
    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        """
        Perform a forward pass of the iPEPS tensor network using the CTM algorithm.

        Args:
            N (int): Number of iterations in the CTM algorithm.
            grad (bool): Whether to compute gradients.
            tensors (tuple): Tuple containing the tensors needed to perform the CTM algorithm.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        pass

    @abstractmethod
    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute and set the energy of the iPEPS tensor network.

        args:
            grad (bool): Whether to compute gradients.
            tensors (tuple): Tuple containing the tensors needed to perform the CTM algorithm.

        Returns:
            torch.Tensor: Energy of the iPEPS tensor network.
        """
        pass


class RotationalSymmetricIPEPS(iPEPS):
    def __init__(self, args: dict, initial_ipeps: "iPEPS" = None):
        super().__init__(args, initial_ipeps)
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()

    def _setup_random(self):
        A = self.tensors.A_random_symmetric(self.args["D"])
        params, self.map = torch.unique(A, return_inverse=True)
        self.params = torch.nn.Parameter(params)

    def _setup_from_initial_ipeps(self):
        self.data = self.initial_ipeps.data
        params = self.initial_ipeps.params.detach()
        self.params = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"])
        self.map = self.initial_ipeps.map

    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        A = self.params[self.map] if self.map is not None else self.params
        A = A.detach() if not grad else A
        C, T = tensors
        E_nn = self.tensors.E_nn(A, self.H, C, T)
        if self.args["model"] == "J1J2":
            return E_nn + self.args["J2"] * self.tensors.E_nnn(A, C, T)
        return E_nn

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        A = self.params[self.map]
        A = A / A.norm()
        A = A.detach() if not grad else A
        iterative = not grad
        alg = CtmSymmetric(A, self.args["chi"], tensors, self.args["split"], iterative)
        alg.exe(N)
        return alg.C, alg.T


class GeneralIPEPS(iPEPS):
    def __init__(self, args: dict, initial_ipeps: "iPEPS" = None):
        super().__init__(args, initial_ipeps)
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()

    def _setup_random(self):
        if self.args["rotational_symmetry"] == "state":
            A = self.tensors.A_random_symmetric(self.args["D"])
        else:
            A = self.tensors.A_random(self.args["D"])
        self.params = torch.nn.Parameter(A / A.norm())

    def _setup_from_initial_ipeps(self):
        self.data = self.initial_ipeps.data
        params = self.initial_ipeps.params.detach()
        if self.initial_ipeps.map is not None:
            # Case where the initial iPEPS is a symmetric state
            params = params[self.initial_ipeps.map]
        self.params = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"])

    def plant_gauge(self):
        A = self.params[self.map] if self.map is not None else self.params
        A = self.tensors.A_gauged(A, which=self.args["gauge"])
        self.params = torch.nn.Parameter(A / A.norm())

    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        A = self.params.detach() if not grad else self.params

        C, T = tensors[:4], tensors[4:]
        E_h = self.tensors.E_horizontal_nn_general(A, self.H, *C, *T)
        E_v = self.tensors.E_vertical_nn_general(A, self.H, *C, *T)
        E_nn = (E_h + E_v) / 2

        if self.args["model"] == "J1J2":
            E_nnn = self.tensors.E_nnn_general(A, *C, *T)
            return E_nn + self.args["J2"] * E_nnn
        return E_nn

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        A = self.params
        A = A / A.norm()
        A = A.detach() if not grad else A
        alg = CtmGeneral(A, self.args["chi"], tensors)
        alg.exe(N)
        return alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
