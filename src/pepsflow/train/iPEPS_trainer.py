from pepsflow.models.tensors import Tensors, Methods
from pepsflow.train.iPEPS import iPEPS
from pepsflow.models.CTM_alg import CtmAlg

import torch
import os
from rich.progress import Progress


class iPEPSTrainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        args (dict): Dictionary containing the arguments for training the iPEPS model.
    """

    def __init__(self, args: dict):
        self.args = args
        torch.set_num_threads(args["threads"])
        self.device = torch.device(
            "cuda:0" if args["gpu"] and torch.cuda.is_available() else "cpu"
        )
        self.data = None
        self.data_prev = (
            torch.load(args["data_fn"], map_location=self.device, weights_only=False)
            if args["data_fn"]
            else None
        )
        self._init_pauli_operators()

    def _init_pauli_operators(self):
        self.sx = torch.Tensor([[0, 1], [1, 0]]).double().to(self.device)
        self.sz = torch.Tensor([[1, 0], [0, -1]]).double().to(self.device)
        self.sy = torch.Tensor([[0, -1], [1, 0]]).double().to(self.device)
        self.sy = torch.complex(torch.zeros_like(self.sz), self.sy).to(self.device)
        self.sp = torch.Tensor([[0, 1], [0, 0]]).double().to(self.device)
        self.sm = torch.Tensor([[0, 0], [1, 0]]).double().to(self.device)
        self.I = torch.eye(2).double().to(self.device)

    def exe(self) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.
        """
        best_model = None

        with Progress() as progress:
            param = self.args["var_param"]
            task = progress.add_task(
                f"[red]Training iPEPS ({param} = {self.args[param]})",
                total=self.args["runs"] * self.args["epochs"],
            )
            for _ in range(self.args["runs"]):
                model = self._train_model(progress, task)

                # Update the best model based on the lowest energy
                if not best_model or model.losses[-1] < best_model.losses[-1]:
                    best_model = model

        self.data = best_model

    def _train_model(self, progress: Progress, task) -> iPEPS:
        """
        Train the iPEPS model for the given parameters.
        """
        checkpoint, map, losses = self._get_checkpoint()
        H = self._get_hamiltonian()
        chi, lam, lr = self.args["chi"], self.args["lam"], self.args["learning_rate"]
        epoch = self.args["start_epoch"] if self.data_prev else -1

        # Perturb the parameters with the given perturbation
        params = checkpoint["params"][epoch]
        params = Methods.perturb(params, self.args["perturbation"])
        checkpoint["params"][epoch] = params

        model = iPEPS(chi, lam, H, params, map).to(self.device)

        # If previous data was given, we start from the last checkpoint
        model.checkpoints = checkpoint
        model.losses = losses[: epoch + 1] if epoch != -1 else losses
        C, T = model.checkpoints["C"][epoch], model.checkpoints["T"][epoch]

        # Initialize the optimizer
        ls = "strong_wolfe" if self.args["line_search"] else None
        optimizer = torch.optim.LBFGS(model.parameters(), lr, 1, line_search_fn=ls)

        def train() -> torch.Tensor:
            """
            Do one step in the CTM algorithm, compute the loss, and do the
            backward pass where the gradients are computed.
            """
            nonlocal C, T
            optimizer.zero_grad()
            loss, C, T = model.forward(C, T)
            loss.backward()
            C, T = C.detach(), T.detach()
            return loss

        for i in range(self.args["epochs"]):
            loss = optimizer.step(train)
            model.losses.append(loss.item())
            # Update the progress bar
            progress.update(task, advance=1)

        # Save the final corner and edge tensors
        model.C = C
        model.T = T

        return model

    def _get_checkpoint(self) -> dict:
        """
        Get the checkpoint dictionary, the map of the parameters for the iPEPS model and the losses.
        This is either initialized from the given data or generated randomly. The checkpoint
        dictionary contains the corner and edge tensors and the parameters.

        Returns:
            tuple: Checkpoint dictionary, the map of the parameters, and the losses.
        """
        # Use the corresponding state from the given data as the initial state.
        if self.data_prev:
            checkpoint = self.data_prev.checkpoints
            losses = self.data_prev.losses
            map = self.data_prev.map
        # Generate a random symmetric A tensor and do CTM warmup steps
        else:
            A = Tensors.A_random_symmetric(self.args["d"]).to(self.device)
            params, map = torch.unique(A, return_inverse=True)
            alg = CtmAlg(a=Tensors.a(A), chi=self.args["chi"])
            alg.exe(max_steps=self.args["warmup_steps"])
            C, T = alg.C, alg.T
            losses = []
            checkpoint = {"C": [C], "T": [T], "params": [params]}

        return checkpoint, map, losses

    def _get_hamiltonian(self) -> torch.Tensor:
        """
        Get the Hamiltonian operator for the iPEPS model.

        Returns:
            torch.Tensor: Hamiltonian operator
        """
        if self.args["model"] == "Heisenberg":
            H = Tensors.H_Heisenberg(
                self.args["lam"], self.sy, self.sz, self.sp, self.sm
            ).to(self.device)
        elif self.args["model"] == "Ising":
            H = Tensors.H_Ising(self.args["lam"], self.sz, self.sx, self.I).to(
                self.device
            )
        else:
            raise ValueError("Invalid model type. Choose 'Heisenberg' or 'Ising'.")
        return H

    def save_data(self, fn: str = None) -> None:
        """
        Save the collected data to a pickle file. The data is saved in the
        'data' directory.

        Args:
            fn (str): Filename to save the data to. Default is 'data.pth'.
        """
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        fn = f"{fn}" if fn else "data.pth"
        torch.save(self.data, fn)
        print(f"Data saved to {fn}")
