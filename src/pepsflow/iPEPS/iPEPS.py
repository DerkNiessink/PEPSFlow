import torch

from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Methods
from pepsflow.models.observables import Observables


class iPEPS(torch.nn.Module):
    """
    Class implementing an infinite Projected Entangled Pair State (iPEPS) tensor network.

    Args:
        chi (int): Bond dimension of the edge and corner tensors.
        lam (float): Regularization parameter for the iPEPS tensor.
        H (torch.Tensor): Hamiltonian operator for the system.
        map (torch.Tensor): indices of the parameters to map to the iPEPS tensor.
        checkpoints (dict): Dictionary containing the corner and edge tensors and the parameters.
        losses (list): List of losses.
        epoch (int): Current epoch.
        perturbation (float): Perturbation value for the parameters.
    """

    def __init__(
        self,
        chi: int,
        split: bool,
        lam: float,
        H: torch.Tensor,
        map: torch.Tensor,
        checkpoints: dict[list, list, list],  # {"C": [], "T": [], "params": []}
        losses: list,
        epoch: int,
        perturbation: float,
        norms: list,
    ):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.split = split
        self.lam = lam
        self.H = H
        self.map = map
        self.checkpoints = checkpoints
        self.losses = losses[: epoch + 1] if epoch != -1 else losses
        self.gradient_norms = norms
        params = Methods.perturb(checkpoints["params"][epoch], perturbation)
        self.params = torch.nn.Parameter(params)
        self.C = checkpoints["C"][epoch]
        self.T = checkpoints["T"][epoch]

    def add_checkpoint(self, C: torch.Tensor, T: torch.Tensor) -> None:
        """
        Add a checkpoint to the dictionary of checkpoints.

        Args:
            C (torch.Tensor): Corner tensor.
            T (torch.Tensor): Edge tensor.
            params (torch.Tensor): Parameters.
        """
        self.checkpoints["C"].append(C.clone().detach())
        self.checkpoints["T"].append(T.clone().detach())
        self.checkpoints["params"].append(self.params.clone().detach())

    def add_loss(self, loss: torch.Tensor) -> None:
        """
        Add the loss to the list of losses.

        Args:
            loss (torch.Tensor): Loss value.
        """
        self.losses.append(loss.item())

    def add_gradient_norm(self) -> None:
        """
        Compute and add the gradient norm to the list of gradient norms.

        Args:
            norm (torch.Tensor): Gradient norm.
        """
        total_norm = torch.sqrt(sum(p.grad.detach().data.norm(2) ** 2 for p in self.parameters()))
        self.gradient_norms.append(total_norm.item())

    def set_edge_corner(self, C: torch.Tensor, T: torch.Tensor) -> None:
        """
        Set the corner and edge tensors of the iPEPS tensor network.

        Args:
            C (torch.Tensor): Corner tensor.
            T (torch.Tensor): Edge tensor.
        """
        self.C = C.detach()
        self.T = T.detach()

    def get_edge_corner(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the corner and edge tensors of the iPEPS tensor network.

        Returns:
            tuple: Corner and edge tensors.
        """
        return self.C.clone().detach(), self.T.clone().detach()

    def set_to_lowest_energy(self) -> None:
        """
        Set the iPEPS tensor network to the state with the lowest energy.
        """
        i = self.losses.index(min(self.losses))

        self.set_edge_corner(self.checkpoints["C"][i], self.checkpoints["T"][i])
        self.params = torch.nn.Parameter(self.checkpoints["params"][i])
        self.losses = self.losses[: i + 1]
        self.gradient_norms = self.gradient_norms[: i + 1]

    def forward(self, C: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Map the parameters to a symmetric rank-5 iPEPS tensor.
        2. Construct the tensor network contraction a.
        3. Execute the CTM (Corner Transfer Matrix) algorithm to compute the new corner (C) and edge (T) tensors.
        4. Compute the loss as the energy expectation value using the Hamiltonian H, the symmetrized tensor,
           and the corner and edge tensors from the CTM algorithm.

        Returns:
            torch.Tensor: The loss, representing the energy expectation value, and the corner and edge tensors.
        """
        # Map the parameters to a symmetric rank-5 iPEPS tensor
        Asymm = self.params[self.map]

        # Do one step of the CTM algorithm
        alg = CtmAlg(Asymm, self.chi, C, T, self.split)
        alg.exe()

        # Compute the energy (loss) using the Hamiltonian, corner, and edge tensors
        loss = Observables.E(Asymm, self.H, alg.C, alg.T)

        return loss, alg.C.detach(), alg.T.detach()
