from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors
from pepsflow.models.observables import Observables
from pepsflow.iPEPS.iPEPS import iPEPS

import torch
import os
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich import print


class Converger:
    """
    Class to compute the converged energies of a PEPS state for different values of chi.

    Args:
        args (dict): Dictionary containing the arguments of the iPEPS model.
    """

    def __init__(self, args: dict):
        data: iPEPS = torch.load(args["data_fn"], weights_only=False)
        self.A = data.params[data.map]
        self.H = Tensors.H_Ising(args["lam"]) if args["model"] == "Ising" else Tensors.H_Heisenberg(args["lam"])
        self.energies = []
        self.args = args

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        self.task = self.progress.add_task(
            f"[blue bold]CTM steps (χ = {self.args['chi']})", total=self.args["warmup_steps"], start=False
        )

    def exe(self):
        """
        Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm.
        """
        self.progress.start_task(self.task)
        alg = CtmAlg(self.A, self.args["chi"], split=self.args["split"])
        for _ in range(self.args["warmup_steps"]):
            alg.exe()
            self.energies.append(Observables.E(self.A, self.H, alg.C, alg.T))
            self.progress.update(self.task, advance=1)

    def save_data(self, fn: str):
        """
        Save the energies to a file.

        Args:
            fn (str): Filename to save the data.
        """
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        fn = f"{fn}" if fn else "data.pth"
        torch.save(self.energies, fn)
        print(f"[green bold] \nData saved to {fn}")
