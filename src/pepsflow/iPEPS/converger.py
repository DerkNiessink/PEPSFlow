import torch

from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors
from pepsflow.models.observables import Observables


class Converger:
    """
    Class to compute the converged energies of a PEPS state for different values of chi.

    Args:
        A (torch.Tensor): Symmetric A tensor of the PEPS state.
        model (str): Model to compute the energy (Ising or Heisenberg).
        lam (float): Parameter of the Hamiltonian.
    """

    def __init__(self, A: torch.Tensor, model: str, lam: float):
        self.A = A
        self.H = Tensors.H_Ising(lam) if model == "Ising" else Tensors.H_Heisenberg(lam)

    def converge(self, chi: int, N: int):
        """
        Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm.

        Args:
            chi (int): Bond dimension of the edge and corner tensors.
            N (int): Number of iterations in the CTMRG algorithm.
        """
        alg = CtmAlg(self.A, chi)
        alg.exe(N=N)
        return Observables.E(self.A, self.H, alg.C, alg.T)
