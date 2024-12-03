import torch

from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Methods, Tensors
from pepsflow.models.observables import Observables


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
        self.args = args
        self.initial_ipeps = initial_ipeps
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()
        self.C, self.T = None, None
        self.data: dict[list, list, list, list, list]

    def _setup_from_initial_ipeps(self) -> None:
        """
        Setup the iPEPS tensor network from the initial data.
        """
        self.data = {}
        epoch = self.args["start_epoch"]
        for key in ["params", "losses", "norms", "C", "T"]:
            self.data[key] = self.initial_ipeps.data[key][: epoch + 1]

        params = Methods.perturb(self.data["params"][epoch], self.args["perturbation"])
        self.map = self.initial_ipeps.map
        self.params = torch.nn.Parameter(params)
        self.H = self.initial_ipeps.H

    def _setup_random(self) -> None:
        """
        Setup the iPEPS tensor network with random parameters.
        """
        A = Tensors.A_random_symmetric(self.args["D"])
        params, self.map = torch.unique(A, return_inverse=True)
        self.data = {"C": [], "T": [], "params": [], "losses": [], "norms": []}
        self.params = torch.nn.Parameter(params)
        self.H = Tensors.Hamiltonian(self.args["model"], lam=self.args["lam"])

    def plant_unitary(self) -> None:
        """
        Plant a unitary matrix on the A tensors of the iPEPS tensor network.
        """
        U = Tensors.random_unitary(self.args["D"])
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
        self.data["C"].append(C)
        self.data["T"].append(T)
        self.C, self.T = C, T

    def set_to_lowest_energy(self) -> None:
        """
        Set the iPEPS tensor network to the state with the lowest energy.
        """
        i = self.data["losses"].index(min(self.data["losses"]))
        self.params = torch.nn.Parameter(self.data["params"][i])
        for key in ["params", "losses", "norms", "C", "T"]:
            self.data[key] = self.data[key][: i + 1]

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Map the parameters to a symmetric rank-5 iPEPS tensor.
        2. Execute the CTM (Corner Transfer Matrix) algorithm to compute the corner (C) and edge (T) tensors.
        3. Compute the loss as the energy expectation value using the Hamiltonian H, the symmetrized tensor,
           and the corner and edge tensors from the CTM algorithm.

        Returns:
            torch.Tensor: The loss, representing the energy expectation value, and the corner and edge tensors.
        """
        A = self.params[self.map]
        A = A / A.norm()

        if torch.isnan(self.params).any():
            raise ValueError("NaN in the iPEPS tensor.")

        alg = CtmAlg(A, self.args["chi"], split=self.args["split"])
        alg.exe(N=self.args["Niter"])

        return Observables.E(A, self.H, alg.C, alg.T), alg.C.detach(), alg.T.detach()
