from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import os
import sys


class Minimizer:
    """
    Class to minimize the energy of the iPEPS model using automatic differentiation.

    Args:
        ipeps (iPEPS): iPEPS model to optimize.
        args (dict): Dictionary containing the arguments for the optimization process.
    """

    def __init__(self, ipeps: iPEPS, args: dict):
        torch.set_num_threads(args["threads"])
        self.args = args
        self.ipeps = ipeps
        ls = "strong_wolfe" if self.args["line_search"] else None
        self.opt = Optimizer(
            self.args["optimizer"], self.ipeps.parameters(), lr=self.args["learning_rate"], line_search_fn=ls
        )

    def minimize(self) -> None:
        """
        Minimize the energy of the iPEPS model using the CTM algorithm and the given optimizer.
        """

        def train() -> torch.Tensor:
            self.opt.zero_grad()
            C_warmup, T_warmup = self.ipeps.do_warmup_steps()
            C, T = self.ipeps.do_gradient_steps(C_warmup, T_warmup)
            loss = self.ipeps.get_E(C, T, grad=True)
            loss.backward()
            return loss

        loss = 0
        for epoch in range(self.args["epochs"]):

            new_loss: torch.Tensor = self.opt.step(train)
            sys.stdout.flush()
            print(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")
            self.ipeps.add_data(new_loss.item())

            if abs(new_loss - loss) < 1e-10:
                sys.stdout.flush()
                print(f"Converged after {epoch} epochs. Saving and quiting training...")
                break
            loss = new_loss

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
        sys.stdout.flush()
        print(f"Data saved to {fn}")
