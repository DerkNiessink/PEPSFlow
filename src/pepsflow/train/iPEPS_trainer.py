from pepsflow.models.tensors import Tensors, Methods
from pepsflow.train.iPEPS import iPEPS

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
        self.init_pauli_operators()

    def init_pauli_operators(self):
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
                # Catch the linalg error and retry with other random tensors
                succes = False
                while not succes:
                    try:
                        model = self._train_model(progress, task)
                    except torch._C._LinAlgError:
                        continue
                    succes = True

                # Update the best model based on the lowest energy
                if not best_model or model.losses[-1] < best_model.losses[-1]:
                    best_model = model

        self.data = best_model

    def _train_model(self, progress: Progress, task) -> iPEPS:
        """
        Train the iPEPS model for the given parameters.
        """
        params, map, H, losses = self._init_tensors()

        # Initialize the iPEPS model and optimizer
        model = iPEPS(self.args["chi"], self.args["lam"], H, params, map, losses).to(
            self.device
        )
        model.train()
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=self.args["learning_rate"], max_iter=1
        )

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _ = model.forward()
            loss.backward()
            return loss

        for _ in range(self.args["epochs"]):
            loss = optimizer.step(train)
            with torch.no_grad():
                model.params.data.clamp_(min=-1, max=1)

            model.losses.append(loss.item())
            # Update the progress bar
            progress.update(task, advance=1)

        # Save the final state of the model
        with torch.no_grad():
            model.losses.append(model.forward()[0].item())

        return model

    def _init_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize the tensors for the iPEPS model.

        Returns:
            torch.Tensor: Parameters of the iPEPS model.
            torch.Tensor: Mapping of the parameters to the iPEPS tensor.
            torch.Tensor: Hamiltonian operator.
            torch.Tensor: Corner tensor for the CTM algorithm.
            torch.Tensor: Edge tensor for the CTM algorithm.
        """
        if self.args["model"] == "Heisenberg":
            H = Tensors.H_Heisenberg(
                lam=self.args["lam"],
                sy=self.sy,
                sz=self.sz,
                sp=self.sp,
                sm=self.sm,
            ).to(self.device)
        elif self.args["model"] == "Ising":
            H = Tensors.H_Ising(
                lam=self.args["lam"], sz=self.sz, sx=self.sx, I=self.I
            ).to(self.device)
        else:
            raise ValueError("Invalid model type. Choose 'Heisenberg' or 'Ising'.")

        losses = []
        # Use the corresponding state from the given data as the initial state.
        if self.data_prev:
            params, map = self.data_prev.params, self.data_prev.map
            losses = self.data_prev.losses
        # Generate a random symmetric A tensor.
        else:
            A = Tensors.A_random_symmetric(self.args["d"]).to(self.device)
            params, map = torch.unique(A, return_inverse=True)

        params = Methods.perturb(params, self.args["perturbation"])
        return params, map, H, losses

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
