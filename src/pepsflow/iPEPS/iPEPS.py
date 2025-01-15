import torch
import numpy as np

from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Methods, Tensors


class iPEPS(torch.nn.Module):
    """
    Class implementing an infinite Projected Entangled Pair State (iPEPS) tensor network.

    Args:
        args (dict): Dictionary containing the arguments for the iPEPS model.
        start_ipeps (dict): iPEPS tensor network to start from.
    """

    def __init__(
        self,
        args: dict,
        initial_ipeps: "iPEPS" = None,
    ):
        super(iPEPS, self).__init__()
        self.to(args["device"])
        if args["seed"] is not None:
            torch.manual_seed(args["seed"])
            np.random.seed(args["seed"])
        self.args = args
        self.initial_ipeps = initial_ipeps
        self.tensors = Tensors(args["dtype"], args["device"])
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()

    def _setup_from_initial_ipeps(self):
        """
        Setup the iPEPS tensor network from the initial data.
        """
        self.data = self.initial_ipeps.data
        params = Methods.perturb(self.initial_ipeps.params.detach(), self.args["perturbation"])
        self.params = torch.nn.Parameter(params)
        self.map = self.initial_ipeps.map
        self.H = self.initial_ipeps.H

    def _setup_random(self):
        """
        Setup the iPEPS tensor network with random parameters.
        """
        A = self.tensors.A_random_symmetric(self.args["D"])
        params, self.map = torch.unique(A, return_inverse=True)
        self.data = {"losses": [], "norms": [], "Niter_warmup": []}
        self.params = torch.nn.Parameter(params)
        self.H = self.tensors.Hamiltonian(self.args["model"], lam=self.args["lam"])

    def get_E(self, C: torch.Tensor, T: torch.Tensor, grad: bool) -> torch.Tensor:
        """
        Compute and set the energy of the iPEPS tensor network. Here we take the next-nearest-neighbor
        (nnn) interaction into account for the J1-J2 model.
        """
        A = self.params[self.map]
        A = A.detach() if not grad else A
        E_nn = self.tensors.E_nn(A, self.H, C, T)
        if self.args["model"] == "J1J2":
            E = E_nn + self.args["J2"] * self.tensors.E_nnn(A, C, T)
        else:
            E = E_nn
        return E

    def _forward(
        self, N: int, grad: bool, C: torch.Tensor = None, T: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Map the parameters to a symmetric rank-5 iPEPS tensor.
        2. Execute the CTM (Corner Transfer Matrix) algorithm to compute the corner (C) and edge (T) tensors.

        Args:
            N (int): Number of CTM steps to perform.
            grad (bool): Whether to compute the gradients for the parameters or not.
            C (torch.Tensor): Corner tensor of the iPEPS tensor network.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network.
        """
        A = self.params[self.map]
        A = A / A.norm()
        # Set requires_grad based on the grad argument
        A = A.detach() if not grad else A
        # Use iterative methods for the eigenvalue decomposition if not computing gradients
        iterative = False if grad else True
        alg = CtmAlg(A, self.args["chi"], C, T, self.args["split"], iterative)
        alg.exe(N)
        return alg.C, alg.T

    def do_warmup_steps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warmup the iPEPS tensor network by performing the CTM algorithm.

        Returns:
            C (torch.Tensor): Corner tensor of the iPEPS tensor network, wich does not require gradients.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network, wich does not require gradients.
        """
        C, T = self._forward(N=self.args["warmup_steps"], grad=False)
        return C, T

    def do_gradient_steps(self, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Take gradient steps in the optimization of the iPEPS tensor network.

        Returns:
            C (torch.Tensor): Corner tensor of the iPEPS tensor network, wich requires gradients.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network, wich requires gradients.
        """
        C, T = self._forward(N=self.args["Niter"], grad=True, C=C, T=T)
        return C, T

    def do_evaluation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the iPEPS tensor network by performing the CTM algorithm.

        Returns:
            C (torch.Tensor): Corner tensor of the iPEPS tensor network, wich does not require gradients.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network, wich does not require gradients.
        """
        C, T = self._forward(N=self.args["Niter"], grad=False)
        return C, T

    def plant_unitary(self):
        """
        Plant a unitary matrix on the A tensors of the iPEPS tensor network.
        """
        U = self.tensors.random_unitary(self.args["D"])
        A = self.params[self.map]
        A = torch.einsum("abcde,bf,cg,dh,ei->afghi", A, U, U, U, U)
        params, self.map = torch.unique(A, return_inverse=True)
        self.params = torch.nn.Parameter(params)

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
        self.data["Niter_warmup"].append(Niter_warmup)
