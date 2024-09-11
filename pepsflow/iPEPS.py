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
        A (torch.Tensor): Rank-5 tensor A with legs of size d, which has left-right,
            up-down, and diagonal symmetry. The legs are ordered as follows: A(phy, up, left, down, right).
            If None, a random symmetric tensor is randomly generated.
    """

    def __init__(
        self,
        chi: int,
        d: int,
        H: torch.Tensor,
        Mpx: torch.Tensor,
        Mpy: torch.Tensor,
        Mpz: torch.Tensor,
        A: torch.Tensor = None,
    ):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.d = d
        self.H = H
        self.Mpx = Mpx
        self.Mpy = Mpy
        self.Mpz = Mpz

        # Initialize the rank-5 tensor A with random values and normalize it.
        # Tensor A is structured as A(phy, up, left, down, right).
        if A is None:
            A = torch.rand(d, d, d, d, d, device=H.device).double()
            A = A / A.norm()
        # If A is provided, convert it to a torch tensor.
        else:
            self.A = A.to(H.device)

        self.A = torch.nn.Parameter(A)

    def forward(self):
        """
        Compute the energy of the iPEPS tensor network by performing the following steps:
        1. Symmetrize the rank-5 tensor A.
        2. Construct the tensor network contraction a.
        3. Execute the CTM (Corner Transfer Matrix) algorithm to compute the new corner (C) and edge (T) tensors.
        4. Compute the loss as the energy expectation value using the Hamiltonian H, the symmetrized tensor,
           and the corner and edge tensors from the CTM algorithm.

        Returns:
            torch.Tensor: The loss, representing the energy expectation value.
        """
        # Symmetrize the rank-5 tensor A
        Asymm = Methods.symmetrize_rank5(self.A)

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
        alg = CtmAlg(a, chi, d)
        alg.exe()

        # Compute the energy (loss) using the Hamiltonian, corner, and edge tensors
        loss, Mx, My, Mz = get_energy(
            Asymm, self.H, alg.C, alg.T, self.Mpx, self.Mpy, self.Mpz
        )

        return loss, Mx, My, Mz
