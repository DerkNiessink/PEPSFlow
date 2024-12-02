from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import os
from rich.progress import Progress, TaskID
from rich import print


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
        torch.random.manual_seed(args["seed"]) if args["seed"] else None
        self.args = args
        self.ipeps = ipeps
        ls = "strong_wolfe" if self.args["line_search"] else None
        self.opt = Optimizer(
            self.args["optimizer"], self.ipeps.parameters(), lr=self.args["learning_rate"], line_search_fn=ls
        )

    def exe(self, progress: Progress, task: TaskID) -> None:
        """
        Optimize the iPEPS model using the CTM algorithm and the given optimizer.

        Args:
            progress (Progress): Rich Progress object to track the progress of the optimization.
            task (TaskID): Task ID of the progress object.
        """

        def train() -> torch.Tensor:
            """
            Do one step in the CTM algorithm, compute the loss, and do the
            backward pass where the gradients are computed.
            """
            self.opt.zero_grad()
            loss, _, _ = self.ipeps.forward()
            loss.backward()
            return loss

        with progress:
            progress.start_task(task)
            for epoch in range(self.args["epochs"]):
                try:
                    self.opt.step(train)

                    with torch.no_grad():
                        loss, C, T = self.ipeps.forward()
                        print(f"epoch, E: {epoch, loss.item()}")

                    # Save intermediate results
                    self.ipeps.add_data(loss, C, T)
                    progress.update(task, advance=1)

                except ValueError:
                    print("[red bold] NaN in iPEPS tensor detected. Saving and quiting training...")
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
        print(f"[green bold] \nData saved to {fn}")
