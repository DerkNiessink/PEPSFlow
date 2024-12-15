from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import os
from rich.progress import Progress, TaskID


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
        self.log = lambda msg: open("log.txt", "a").write(msg + "\n") if self.args["log"] else lambda msg: None

    def exe(self, progress: Progress = None, task: TaskID = None) -> None:
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

        with progress:
            progress.start_task(task) if progress else None
            loss = 0
            for epoch in range(self.args["epochs"]):
                try:
                    new_loss = self.opt.step(train)
                    self.log(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")

                    self.ipeps.add_data(new_loss, C, T)
                    progress.update(task, advance=1) if progress else None

                    if abs(new_loss - loss) < 1e-9:
                        self.log(f"Converged after {epoch} epochs. Saving and quiting training...")
                        break
                    loss = new_loss

                except ValueError:
                    self.log("NaN in iPEPS tensor detected. Saving and quiting training...")
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
        self.log(f"Data saved to {fn}")
