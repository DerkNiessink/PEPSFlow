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
        Mpx (torch.Tensor): X Pauli operator.
        Mpy (torch.Tensor): Y Pauli operator.
        Mpz (torch.Tensor): Z Pauli operator.
        params (torch.Tensor): Parameters to optimize.
        maps (torch.Tensor): indices of the parameters to map to the iPEPS tensor.
        C (torch.Tensor): Initial corner tensor for the CTM algorithm.
        T (torch.Tensor): Initial edge tensor for the CTM algorithm.
    """

    def __init__(
        self,
        chi: int,
        H: torch.Tensor,
        params: torch.Tensor,
        map: torch.Tensor,
        C: torch.Tensor,
        T: torch.Tensor,
    ):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.H = H
        self.params = torch.nn.Parameter(params)
        self.map = map
        self.C = C
        self.T = T
        self.loss = None

    def forward(self) -> torch.Tensor:
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

        # Execute the CTM algorithm to compute corner (C) and edge (T) tensors
        alg = CtmAlg(Tensors.a(Asymm), self.chi, self.C, self.T)
        alg.exe()

        # Compute the energy (loss) using the Hamiltonian, corner, and edge tensors
        self.loss = Observables.E(Asymm, self.H, alg.C, alg.T)

        return self.loss, alg.C, alg.T
