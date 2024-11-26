from pepsflow.models.tensors import Tensors
from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.optimizers import Optimizer

import torch
import os
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich import print


class Trainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        args (dict): Dictionary containing the arguments for training the iPEPS model.
    """

    def __init__(self, args: dict):
        torch.set_num_threads(args["threads"])
        self.args = args
        self.data, self.data_prev = None, None

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
        param = self.args["var_param"]
        self.warmup_task = self.progress.add_task(
            f"[blue bold]CTM steps ({param} = {self.args[param]})",
            total=self.args["warmup_steps"] * self.args["runs"],
            start=False,
            visible=False,
        )
        self.training_task = self.progress.add_task(
            f"[blue bold]Training iPEPS ({param} = {self.args[param]})",
            total=self.args["runs"] * self.args["epochs"],
            start=False,
        )

    def read(self, fn: str) -> None:
        """
        Read the data from a torch .pth file.

        Args:
            fn (str): Filename to read the data from, without file extension. Has to be a .pth file.
        """
        self.data_prev: iPEPS = torch.load(fn, weights_only=False)

    def exe(self) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.
        """
        best_model = None

        with self.progress:
            for _ in range(self.args["runs"]):
                model = self._train_model()

                # Update the best model based on the lowest energy
                if not best_model or model.losses[-1] < best_model.losses[-1]:
                    best_model = model

        self.data = best_model

    def _train_model(self) -> iPEPS:
        """
        Train the iPEPS model for the given parameters.
        """

        checkpoint, map, losses, epoch, norms = self._get_checkpoint()

        H = (
            Tensors.H_Heisenberg(self.args["lam"])
            if self.args["model"] == "Heisenberg"
            else Tensors.H_Ising(self.args["lam"])
        )
        chi, lam, per, split = self.args["chi"], self.args["lam"], self.args["perturbation"], self.args["split"]
        model = iPEPS(chi, split, lam, H, map, checkpoint, losses, epoch, per, norms)

        optimizer = Optimizer(
            self.args["optimizer"],
            model.parameters(),
            lr=self.args["learning_rate"],
            line_search_fn=self.args["line_search"],
        )

        C, T = model.get_edge_corner()

        def train() -> torch.Tensor:
            """
            Do one step in the CTM algorithm, compute the loss, and do the
            backward pass where the gradients are computed.
            """
            nonlocal C, T
            optimizer.zero_grad()
            loss, C, T = model.forward(C, T)
            loss.backward()
            return loss

        self.progress.start_task(self.training_task)
        for _ in range(self.args["epochs"]):
            loss = optimizer.step(train)

            # Save intermediate results
            model.add_checkpoint(C, T)
            model.add_loss(loss)
            model.add_gradient_norm()

            # Update the progress bar
            self.progress.update(self.training_task, advance=1)

        # Save the final corner and edge tensors
        model.set_edge_corner(C, T)

        return model

    def _get_checkpoint(self) -> dict:
        """
        Get the checkpoint dictionary, the map of the parameters for the iPEPS model and the losses.
        This is either initialized from the given data or generated randomly. The checkpoint
        dictionary contains the corner and edge tensors and the parameters.

        Returns:
            tuple: Checkpoint dictionary, the map of the parameters, losses, the current epoch, and
            the gradient norms.
        """
        # Use the corresponding state from the given data as the initial state.
        if self.data_prev:
            checkpoint = self.data_prev.checkpoints
            losses = self.data_prev.losses
            gradient_norms = self.data_prev.gradient_norms
            map = self.data_prev.map
            epoch = self.args["start_epoch"]

        # Generate a random symmetric A tensor and do CTM warmup steps
        else:
            A = Tensors.A_random_symmetric(self.args["D"])
            params, map = torch.unique(A, return_inverse=True)
            alg = CtmAlg(A, chi=self.args["chi"], split=self.args["split"])
            self.progress.tasks[self.warmup_task].visible = True
            alg.exe(N=self.args["warmup_steps"], progress=self.progress, task=self.warmup_task)
            C, T = alg.C, alg.T
            checkpoint = {"C": [C], "T": [T], "params": [params]}
            losses, gradient_norms = [], []
            epoch = -1

        return checkpoint, map, losses, epoch, gradient_norms

    def write(self, fn: str) -> None:
        """
        Save the collected data to a torch .pth file.

        Args:
            fn (str): Filename to save the data to. Has to be a .pth file.
        """
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.data, fn)
        print(f"[green bold] \nData saved to {fn}")
