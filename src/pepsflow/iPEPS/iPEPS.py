import torch
import numpy as np

from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
from pepsflow.models.tensors import Tensors


class iPEPS(torch.nn.Module):
    """
    Base class implementing an infinite Projected Entangled Pair State (iPEPS) tensor network.

    Args:
        args (dict): Dictionary containing the arguments for the iPEPS model.
        start_ipeps (dict): iPEPS tensor network to start from.
    """

    def __init__(self, args: dict, initial_ipeps: "iPEPS" = None):
        super(iPEPS, self).__init__()
        self.to(args["device"])
        if args["seed"] is not None:
            torch.manual_seed(args["seed"])
            np.random.seed(args["seed"])
        self.args = args
        self.initial_ipeps = initial_ipeps
        self.tensors = Tensors(args["dtype"], args["device"])
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()
        self.U1 = self.tensors.random_tensor(shape=(self.args["D"], self.args["D"]))
        self.U2 = self.tensors.random_tensor(shape=(self.args["D"], self.args["D"]))

    def _setup_from_initial_ipeps(self):
        """
        Setup the iPEPS tensor network from the initial data.
        """
        self.data = self.initial_ipeps.data
        self.H = self.initial_ipeps.H
        if self.args["rotational_symmetry"] in ["both", "state"]:
            params = self.initial_ipeps.params.detach()
            self.params = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"])
            self.map = self.initial_ipeps.map
        elif self.args["rotational_symmetry"] in [None, "ctm"]:
            params = self.initial_ipeps.params.detach()
            # Determine if the initial ipeps is a symmetric state or not
            params = params[self.initial_ipeps.map] if self.initial_ipeps.map is not None else params
            self.params, self.map = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"]), None
        else:
            raise ValueError(f"Unknown rotational_symmetry type: {self.args['rotational_symmetry']}")

    def _setup_random(self):
        """
        Setup the iPEPS tensor network with random parameters.
        """
        self.data = {"losses": [], "norms": [], "Niter_warmup": []}
        self.H = self.tensors.Hamiltonian(self.args["model"], lam=self.args["lam"])

        if self.args["rotational_symmetry"] in ["both", "state"]:
            A = self.tensors.A_random_symmetric(self.args["D"])
            params, self.map = torch.unique(A, return_inverse=True)
            self.params = torch.nn.Parameter(params)
        elif self.args["rotational_symmetry"] in [None, "ctm"]:
            A = self.tensors.A_random(self.args["D"])
            self.params, self.map = torch.nn.Parameter(A / A.norm()), None
        else:
            raise ValueError(f"Unknown rotational_symmetry type: {self.args['rotational_symmetry']}")

    def plant_gauge(self):
        """
        Plant a unitary matrix on the A tensors of the iPEPS tensor network.
        """
        if self.args["gauge"] == "unitary":
            U1 = self.tensors.random_unitary(self.args["D"])
            U2 = self.tensors.random_unitary(self.args["D"])
        elif self.args["gauge"] == "invertible":
            U1 = self.tensors.random_tensor(shape=(self.args["D"], self.args["D"]))
            U2 = self.tensors.random_tensor(shape=(self.args["D"], self.args["D"]))
        elif self.args["gauge"] == None:
            U1 = U2 = self.tensors.identity(self.args["D"])
        else:
            raise ValueError(f"Unknown gauge type: {self.args['gauge']}")

        A = self.params[self.map] if self.map is not None else self.params
        A = torch.einsum("abcde,bf,cg,dh,ei->afghi", A, U1, torch.linalg.inv(U2).T, torch.linalg.inv(U1).T, U2)
        #         |
        #        |_|
        #    _    |    _                /
        # --|_|-- o --|_|--    🡺   -- o --
        #         | \                 /|
        #        |_| \
        #         |
        self.params = torch.nn.Parameter(A / A.norm())

    def add_data(self, E: torch.Tensor = None, Niter_warmup: int = None):
        """
        Add data of the energy, norm and number of warmup iterations to the iPEPS tensor network.

        Args:
            E (torch.Tensor): Energy of the iPEPS tensor network.
            Niter_warmup (int): Number of warmup iterations.
        """
        self.data["losses"].append(E)
        squared_norm = sum(p.data.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
        self.data["norms"].append(torch.sqrt(squared_norm) if isinstance(squared_norm, torch.Tensor) else squared_norm)
        # self.data["Niter_warmup"].append(Niter_warmup)

    def do_warmup_steps(self) -> tuple[torch.Tensor, ...]:
        """
        Warmup the iPEPS tensor network by performing the CTM algorithm without gradient tracking.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["warmup_steps"], grad=False)

    def do_gradient_steps(self, tensors) -> tuple[torch.Tensor, ...]:
        """
        Take gradient steps in the optimization of the iPEPS tensor network.

        Args:
            tensors (tuple[torch.Tensor, ...]): Tuple containing the tensors needed to perform the CTM algorithm.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["Niter"], grad=True, tensors=tensors)

    def do_evaluation(self) -> tuple[torch.Tensor, ...]:
        """
        Evaluate the iPEPS tensor network by performing the CTM algorithm without gradient tracking.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N=self.args["Niter"], grad=False)

    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute and set the energy of the iPEPS tensor network. Here we take the next-nearest-neighbor
        (nnn) interaction into account for the J1-J2 model.
        """
        A = self.params[self.map] if self.map is not None else self.params
        A = A.detach() if not grad else A

        if self.args["rotational_symmetry"] in ["both", "ctm"]:
            C, T = tensors
            E_nn = self.tensors.E_nn(A, self.H, C, T)
            if self.args["model"] == "J1J2":
                E = E_nn + self.args["J2"] * self.tensors.E_nnn(A, C, T)
            else:
                E = E_nn

        else:
            C1, C2, C3, C4, T1, T2, T3, T4 = tensors
            E_horizontal_nn = self.tensors.E_horizontal_nn_general(A, self.H, C1, C2, C3, C4, T1, T2, T3, T4)
            E_vertical_nn = self.tensors.E_vertical_nn_general(A, self.H, C1, C2, C3, C4, T1, T2, T3, T4)
            E_nn = (E_horizontal_nn + E_vertical_nn) / 2

            if self.args["model"] == "J1J2":
                E = E_nn + self.args["J2"] * self.tensors.E_nnn_general(A, C1, C2, C3, C4, T1, T2, T3, T4)
            else:
                E = E_nn
        return E

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple[torch.Tensor, ...]:
        """
        Perform a forward pass of the iPEPS tensor network using the CTM algorithm.

        args:
            N (int): Number of iterations in the CTM algorithm.
            grad (bool): Whether to compute gradients.
            tensors (tuple): Tuple containing the tensors needed to perform the CTM algorithm.

        return tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        A = self.params[self.map] if self.map is not None else self.params

        # A = torch.einsum(
        #    "abcde,bf,cg,dh,ei->afghi", A, self.U1, torch.linalg.inv(self.U2).T, torch.linalg.inv(self.U1).T, self.U2
        # )

        A = A / A.norm()
        # Set requires_grad based on the grad argument
        A = A.detach() if not grad else A

        # Use iterative methods for the eigenvalue decomposition if not computing gradients
        iterative = False if grad else True

        if self.args["rotational_symmetry"] in ["both", "ctm"]:
            alg = CtmSymmetric(A, self.args["chi"], tensors, self.args["split"], iterative)
            alg.exe(N)
            return alg.C, alg.T
        else:
            alg = CtmGeneral(A, self.args["chi"], tensors, iterative=iterative)
            alg.exe(N)

            return alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
