from pepsflow.models.tensors import Tensors, Methods
from pepsflow.train.iPEPS import iPEPS

import torch
import os
from tqdm import tqdm


class iPEPSTrainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        args (dict): Dictionary containing the arguments for training the iPEPS model.
    """

    def __init__(self, args: dict):
        self.args = args
        self.device = torch.device(
            "cuda:0" if args["gpu"] and torch.cuda.is_available() else "cpu"
        )
        self.sx = torch.Tensor([[0, 1], [1, 0]]).double().to(self.device)
        self.sz = torch.Tensor([[1, 0], [0, -1]]).double().to(self.device)
        self.I = torch.eye(2).double().to(self.device)
        self.data = None
        self.data_prev = (
            torch.load(args["data_fn"], map_location=self.device, weights_only=False)
            if args["data_fn"]
            else None
        )

    def exe(self) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.
        """
        best_model = None
        for _ in range(self.args["runs"]):
            # Catch the linalg error and retry with other random tensors
            succes = False
            while not succes:
                try:
                    model = self._train_model()
                except torch._C._LinAlgError:
                    continue
                succes = True

            # Update the best model based on the lowest energy
            if not best_model or model.loss < best_model.loss:
                best_model = model
        self.data = best_model

    def _train_model(self) -> iPEPS:
        """
        Train the iPEPS model for the given parameters.
        """
        params, map, H, C, T = self._init_tensors()

        # Initialize the iPEPS model and optimizer
        model = iPEPS(self.args["chi"], H, params, map, C, T).to(self.device)
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=self.args["lr"], max_iter=self.args["max_iter"]
        )

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _ = model.forward()
            loss.backward()
            return loss

        for _ in tqdm(range(self.args["epochs"])):
            optimizer.step(train)

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
        H = Tensors.H(lam=self.args["lam"], sz=self.sz, sx=self.sx, I=self.I)
        C = T = None

        # Use the corresponding state from the given data as the initial state.
        if self.data_prev:
            params, map = self.data_prev.params, self.data_prev.map
        # Generate a random symmetric A tensor.
        else:
            A = Tensors.A_random_symmetric(self.args["D"]).to(self.device)
            params, map = torch.unique(A, return_inverse=True)

        params = Methods.perturb(params, self.args["perturbation"])
        return params, map, H, C, T

    def save_data(self, fn: str = None) -> None:
        """
        Save the collected data to a pickle file. The data is saved in the
        'data' directory.

        Args:
            fn (str): Filename to save the data to. Default is 'data.pth'.
        """
        directory = "data"
        filename = f"{fn}.pth" if fn else "data.pth"

        # Ensure the directory structure exists
        full_path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        torch.save(self.data, full_path)
        print(f"Data saved to {full_path}")
