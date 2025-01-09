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
        self.C, self.T = None, None
        self.Niter_warmup = None
        self.data: dict[list, list, list, list, list]

        self.tensors = Tensors(args["dtype"], args["device"])
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()

    def _setup_from_initial_ipeps(self) -> None:
        """
        Setup the iPEPS tensor network from the initial data.
        """
        self.data = {}
        epoch = self.args["start_epoch"]

        # Copy the data from the initial iPEPS tensor network and handle the case when the epoch is -1
        for key in ["params", "losses", "norms", "C", "T", "Niter_warmup"]:
            self.data[key] = self.initial_ipeps.data[key][: epoch + 1] if epoch != -1 else self.initial_ipeps.data[key]
        if epoch == -1:
            epoch = len(self.data["params"]) - 1

        params = Methods.perturb(self.data["params"][epoch], self.args["perturbation"])
        self.map = self.initial_ipeps.map
        self.params = torch.nn.Parameter(params)
        self.H = self.initial_ipeps.H

    def _setup_random(self) -> None:
        """
        Setup the iPEPS tensor network with random parameters.
        """
        A = self.tensors.A_random_symmetric(self.args["D"])
        params, self.map = torch.unique(A, return_inverse=True)
        self.data = {"C": [], "T": [], "params": [], "losses": [], "norms": [], "Niter_warmup": []}
        self.params = torch.nn.Parameter(params)
        self.H = self.tensors.Hamiltonian(self.args["model"], lam=self.args["lam"])

    def plant_unitary(self) -> None:
        """
        Plant a unitary matrix on the A tensors of the iPEPS tensor network.
        """
        U = self.tensors.random_unitary(self.args["D"])
        A = self.params[self.map]
        A = torch.einsum("abcde,bf,cg,dh,ei->afghi", A, U, U, U, U)
        params, self.map = torch.unique(A, return_inverse=True)
        self.params = torch.nn.Parameter(params)

    def add_data(self, loss: torch.Tensor, C: torch.Tensor, T: torch.Tensor) -> None:
        """
        Add the loss, corner, and edge tensors to the data dictionary.

        Args:
            loss (torch.Tensor): Loss value of the optimization step.
            C (torch.Tensor): Corner tensor of the iPEPS tensor network.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network.
        """
        self.data["losses"].append(loss)
        self.data["params"].append(self.params.clone().detach())
        squared_norm = sum(p.data.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
        self.data["norms"].append(torch.sqrt(squared_norm) if isinstance(squared_norm, torch.Tensor) else squared_norm)
        if self.args["save_intermediate"]:  # Do not save the corner and edge tensors to save a lot of memory
            self.data["C"].append(C)
            self.data["T"].append(T)
        self.data["Niter_warmup"].append(self.Niter_warmup)
        self.C, self.T = C, T

    def do_warmup_steps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Warmup the iPEPS tensor network by performing the CTM algorithm.

        Returns:
            tuple: Corner tensor and edge tensor
        """
        _, C, T = self._forward(warmup=True)
        return C, T

    def do_gradient_steps(self, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take gradient steps in the optimization of the iPEPS tensor network.

        Args:
            C (torch.Tensor): Initial corner tensor for the CTM algorithm.
            T (torch.Tensor): Initial edge tensor for the CTM algorithm.

        Returns:
            tuple: Loss, corner tensor, and edge tensor
        """
        loss, C, T = self._forward(self.C, self.T, warmup=False)
        return loss, C, T

    def _get_energy(self, A, C, T) -> torch.Tensor:
        """
        Compute the energy of the iPEPS tensor network. Here we take the next-nearest-neighbor
        (nnn) interaction into account for the J1-J2 model.

        Returns:
            torch.Tensor: Energy of the iPEPS tensor network.
        """
        if self.args["model"] == "J1J2":
            energy = self.tensors.E_nn(A, self.H, C, T) + self.args["J2"] * self.tensors.E_nnn(A, C, T)
        else:
            energy = self.tensors.E_nn(A, self.H, C, T)

        return energy

    def _forward(
        self, C: torch.Tensor = None, T: torch.Tensor = None, warmup: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Map the parameters to a symmetric rank-5 iPEPS tensor.
        2. Execute the CTM (Corner Transfer Matrix) algorithm to compute the corner (C) and edge (T) tensors.
        3. Compute the loss as the energy expectation value using the Hamiltonian H, the symmetrized tensor,
           and the corner and edge tensors from the CTM algorithm.

        Args:
            C (torch.Tensor): Initial corner tensor for the CTM algorithm. Default is None.
            T (torch.Tensor): Initial edge tensor for the CTM algorithm. Default is None.
            warmup (bool): Flag to indicate if the warmup steps should be executed. Default is False.

        Returns:
            torch.Tensor: The loss, representing the energy expectation value, the corner tensor, and edge tensor.
        """
        A = self.params[self.map]
        A = A / A.norm()

        if torch.isnan(self.params).any():
            raise ValueError("NaN in the iPEPS tensor.")

        alg = CtmAlg(A, self.args["chi"], C, T, self.args["split"])

        N = self.args["warmup_steps"] if warmup else self.args["Niter"]
        alg.exe(N)

        # The loss does not have to be computed in the warmup steps
        if warmup:
            loss = None
            self.Niter_warmup = alg.Niter
        else:
            loss = self._get_energy(A, alg.C, alg.T)

        return loss, alg.C.detach(), alg.T.detach()
