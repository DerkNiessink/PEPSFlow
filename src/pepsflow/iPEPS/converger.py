from pepsflow.iPEPS.iPEPS import iPEPS

import torch
import os
from rich import print


class Converger:
    """
    Class to compute the converged energies of a PEPS state for different values of chi.

    Args:
        ipeps (iPEPS): iPEPS model to compute the energies for.
        args (dict): Dictionary containing the iPEPS parameters.
    """

    def __init__(self, ipeps: iPEPS, args: dict):
        self.ipeps = iPEPS(args, ipeps)
        self.ipeps.plant_unitary()

    def exe(self) -> None:
        """
        Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm. This method can only be called after reading the data.
        """
        with torch.no_grad():
            E, C, T = self.ipeps.do_gradient_steps(self.ipeps.C, self.ipeps.T)
        self.ipeps.add_data(E, C, T)
        print(f"chi, E: {self.ipeps.args['chi'], E.item()}")

    def write(self, fn: str) -> None:
        """
        Save the collected data to a torch .pth file.

        Args:
            fn (str): Filename to save the data to.
        """
        fn = f"{fn}.pth"
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.ipeps, fn)
        print(f"[green bold] \nData saved to {fn}")
