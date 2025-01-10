from pepsflow.ipeps.ipeps import iPEPS
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
        C, T = None, None

        def train() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal C, T
            self.opt.zero_grad()
            C, T = self.ipeps.do_warmup_steps()
            loss, C, T = self.ipeps.do_gradient_steps(C, T)
            loss.backward()
            return loss

        loss = 0
        for epoch in range(self.args["epochs"]):
            sys.stdout.flush()
            try:
                new_loss = self.opt.step(train)
                print(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")

                self.ipeps.add_data(new_loss, C, T)

                if abs(new_loss - loss) < 1e-10:
                    sys.stdout.flush()
                    print(f"Converged after {epoch} epochs. Saving and quiting training...")
                    break
                loss = new_loss

            except ValueError:
                sys.stdout.flush()
                print("NaN in iPEPS tensor detected. Saving and quiting training...")
                break

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
