from pepsflow.iPEPS.iPEPS import iPEPS

import torch
import os
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
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

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        self.task = self.progress.add_task(
            f"[blue bold]CTM steps (χ = {args['chi']})", total=args["Niter"], start=False
        )

    def exe(self):
        """
        Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm. This method can only be called after reading the data.
        """
        Niter = self.ipeps.args["Niter"]
        self.ipeps.args["Niter"] = 1
        with self.progress:
            for _ in range(Niter):
                E, C, T = self.ipeps.forward()
                self.ipeps.add_data(E, C, T)
                self.progress.update(self.task, advance=1)

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
