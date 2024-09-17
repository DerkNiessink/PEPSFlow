import torch

from pepsflow.models.tensors import Methods
from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.energy import get_energy


class iPEPS(torch.nn.Module):
    """
    Class implementing an infinite Projected Entangled Pair State (iPEPS) tensor network.

    Args:
        chi (int): Bond dimension of the edge and corner tensors.
        d (int): Physical dimension of the local Hilbert space.
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
        d: int,
        H: torch.Tensor,
        Mpx: torch.Tensor,
        Mpy: torch.Tensor,
        Mpz: torch.Tensor,
        params: torch.Tensor,
        map: torch.Tensor,
        C: torch.Tensor,
        T: torch.Tensor,
    ):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.d = d
        self.H = H
        self.Mpx = Mpx
        self.Mpy = Mpy
        self.Mpz = Mpz
        self.params = torch.nn.Parameter(params)
        self.map = map
        self.C = C
        self.T = T
        self.loss = None

    def forward(self):
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Map the parameters to a symmetric rank-5 iPEPS tensor.
        2. Construct the tensor network contraction a.
        3. Execute the CTM (Corner Transfer Matrix) algorithm to compute the new corner (C) and edge (T) tensors.
        4. Compute the loss as the energy expectation value using the Hamiltonian H, the symmetrized tensor,
           and the corner and edge tensors from the CTM algorithm.

        Returns:
            torch.Tensor: The loss, representing the energy expectation value.
        """
        # Map the parameters to a symmetric rank-5 iPEPS tensor
        Asymm = self.params[self.map]
        d, chi = self.d, self.chi

        # Construct the tensor network contraction a
        a = (
            (Asymm.view(d, -1).t() @ Asymm.view(d, -1))
            .contiguous()
            .view(d, d, d, d, d, d, d, d)
        )
        a = a.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(d**2, d**2, d**2, d**2)
        a = a / a.norm()

        # Execute the CTM algorithm to compute corner (C) and edge (T) tensors
        alg = CtmAlg(a, chi, d, self.C, self.T)
        alg.exe()

        if torch.isnan(a).any():
            raise ValueError("NaNs in the tensor network contraction a")

        # Compute the energy (loss) using the Hamiltonian, corner, and edge tensors
        loss, Mx, My, Mz = get_energy(
            Asymm, self.H, alg.C, alg.T, self.Mpx, self.Mpy, self.Mpz
        )
        self.loss = loss
        return loss, Mx, My, Mz, alg.C, alg.T
