import torch

from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors
from pepsflow.models.observables import Observables


class iPEPS(torch.nn.Module):
    """
    Class implementing an infinite Projected Entangled Pair State (iPEPS) tensor network.

    Args:
        chi (int): Bond dimension of the edge and corner tensors.
        H (torch.Tensor): Hamiltonian operator for the system.
        lam (float): Regularization parameter for the iPEPS tensor.
        params (torch.Tensor): Parameters to optimize.
        map (torch.Tensor): indices of the parameters to map to the iPEPS tensor.
    """

    def __init__(
        self,
        chi: int,
        lam: float,
        H: torch.Tensor,
        params: torch.Tensor,
        map: torch.Tensor,
    ):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.lam = lam
        self.H = H
        self.params = torch.nn.Parameter(params)
        self.map = map
        self.losses = []
        self.C = None
        self.T = None

    def forward(
        self, C: torch.Tensor, T: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        alg = CtmAlg(Tensors.a(Asymm), self.chi, C, T)
        alg.exe(1)

        # Compute the energy (loss) using the Hamiltonian, corner, and edge tensors
        loss = Observables.E(Asymm, self.H, alg.C, alg.T)

        return loss, alg.C, alg.T
