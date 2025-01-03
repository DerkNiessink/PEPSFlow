from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
from datetime import datetime
import os
from rich.progress import Progress, TaskID
import sys


class Trainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        opt_args (dict): Dictionary containing the arguments for the optimization process.
        ipeps_args (dict): Dictionary containing the arguments for the iPEPS model.
    """

    def __init__(self, ipeps: iPEPS, args: dict):
        torch.set_num_threads(args["threads"])
        self.args = args
        self.ipeps = ipeps
        ls = "strong_wolfe" if self.args["line_search"] else None
        self.opt = Optimizer(
            self.args["optimizer"], self.ipeps.parameters(), lr=self.args["learning_rate"], line_search_fn=ls
        )

    def exe(self) -> None:
        """
        Optimize the iPEPS model using the CTM algorithm and the given optimizer.

        Args:
            progress (Progress): Rich Progress object to track the progress of the optimization.
            task (TaskID): Task ID of the progress object.
        """
        C, T = None, None

        def train() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Do one step in the CTM algorithm, compute the loss, and do the
            backward pass where the gradients are computed.
            """
            nonlocal C, T
            self.opt.zero_grad()
            C, T = self.ipeps.warmup()
            loss, C, T = self.ipeps.forward(C, T)
            loss.backward()
            return loss

        loss = 0
        for epoch in range(self.args["epochs"]):
            sys.stdout.flush()
            try:
                new_loss = self.opt.step(train)
                print(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")

                self.ipeps.add_data(new_loss, C, T)

                if abs(new_loss - loss) < 1e-9:
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
