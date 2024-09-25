from pepsflow.models.tensors import Tensors, Methods
from pepsflow.iPEPS import iPEPS

import torch
import os
from tqdm import tqdm


class iPEPSTrainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        chi (int): Bond dimension of the edge and corner tensors.
        d (int): Physical dimension of the local Hilbert space.
        gpu (bool): Whether to run the model on the GPU if available. Default
            is False.
        data_fn (str): Filename of the data to load. Default is None.
    """

    def __init__(self, chi: int, d: int, gpu: bool = False, data_fn: str = None):
        self.chi = chi
        self.d = d
        self.device = torch.device(
            "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.sx = torch.Tensor([[0, 1], [1, 0]]).double().to(self.device)
        self.sz = torch.Tensor([[1, 0], [0, -1]]).double().to(self.device)
        self.I = torch.eye(2).double().to(self.device)
        self.clear_data()
        self.data_prev = (
            torch.load(data_fn, map_location=self.device, weights_only=False)
            if data_fn
            else None
        )

    def exe(
        self,
        lambdas: list = [0.5],
        epochs: int = 10,
        use_prev: bool = False,
        perturbation: float = 0.0,
        runs: int = 1,
        lr: float = 1,
        max_iter: int = 20,
    ) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.

        Args:
            lambdas (list): List of values of lambda.
            epochs (int): Maximum number of epochs to train the model.
            use_prev (bool): Whether to use the previous state as the initial
                state for the training.
            perturbation (float): Amount of perturbation to apply to the initial
                state. Default is 0.0.
            runs (int): Number of runs to train the model. Default is 1.
            lr (float): Learning rate for the optimizer. Default is 1.
            max_iter (int): Maximum number of iterations for the optimizer. Default is 20.
        """
        lambdas = self.data_prev.keys() if self.data_prev else lambdas
        for lam in tqdm(lambdas):

            best_model = None
            for _ in range(runs):

                success = False
                while not success:
                    try:
                        model = self._train_model(
                            lam, epochs, use_prev, perturbation, lr, max_iter
                        )
                        success = True
                    except torch._C._LinAlgError:
                        continue

                if not best_model or model.loss < best_model.loss:
                    best_model = model

            # Save the best model for the given lambda
            self.data[lam] = best_model

    def _train_model(
        self,
        lam: float,
        epochs: int,
        use_prev: bool,
        perturbation: float,
        lr: float,
        max_iter: int,
    ) -> None:
        """
        Train the iPEPS model for a given value of lambda.

        Args:
            lam (float): Value of lambda.
            epochs (int): Maximum number of epochs to train the model.
            n_runs (int): Number of runs to train the model.
            use_prev (bool): Whether to use the previous state as the initial
                state for the training.
            lr (float): Learning rate for the optimizer.
            max_iter (int): Maximum number of iterations for the optimizer.
        """

        params, map, H, C, T = self._init_tensors(lam, use_prev, perturbation)

        # Initialize the iPEPS model and optimizer
        model = iPEPS(self.chi, self.d, H, params, map, C, T).to(self.device)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=max_iter, lr=lr)

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _ = model.forward()
            loss.backward()
            return loss

        for _ in range(epochs):
            optimizer.step(train)

        return model

    def _init_tensors(self, lam: float, use_prev: bool, perturbation: float) -> None:
        """
        Initialize the tensors for the iPEPS model.

        Args:
            lam (float): Value of lambda.
            use_prev (bool): Whether to use the previous state as the initial
                state for the training.
            perturbation (float): Amount of perturbation to apply to the initial
                state. Default is 0.0

        Returns:
            torch.Tensor: Parameters of the iPEPS model.
            torch.Tensor: Mapping of the parameters to the iPEPS tensor.
            torch.Tensor: Hamiltonian operator.
            torch.Tensor: Corner tensor for the CTM algorithm.
            torch.Tensor: Edge tensor for the CTM algorithm.
        """
        H = Tensors.H(lam=lam, sz=self.sz, sx=self.sx, I=self.I)
        C = T = None

        # Use the previous state as the initial state.
        if self.data and use_prev:
            prev_lam = list(self.data.keys())[-1]
            params, map = self.data[prev_lam].params, self.data[prev_lam].map

        # Use the corresponding state from the given data as the initial state.
        elif self.data_prev:
            params, map = self.data_prev[lam].params, self.data_prev[lam].map

        # Generate a random symmetric A tensor.
        else:
            A = Tensors.A_random_symmetric(d=self.d).to(self.device)
            params, map = torch.unique(A, return_inverse=True)

        return Methods.perturb(params, perturbation), map, H, C, T

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

    def clear_data(self) -> None:
        """
        Clear the collected data.
        """
        self.data = {}
