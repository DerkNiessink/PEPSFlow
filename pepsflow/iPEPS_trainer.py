from pepsflow.models.tensors import Tensors, Methods
from pepsflow.iPEPS import iPEPS

import pickle
import torch
import os
from tqdm import tqdm
import time


class iPEPSTrainer:
    """
    Class to train the iPEPS model using automatic differentiation for different
    values of lambda.

    Args:
        chi (int): Bond dimension of the edge and corner tensors.
        d (int): Physical dimension of the local Hilbert space.
        gpu (bool): Whether to run the model on the GPU if available. Default
            is False.
        data (dict): Dictionary of previous data to use as initial state for the
            training. Default is None.
    """

    def __init__(self, chi: int, d: int, gpu: bool = False, data_prev: dict = None):
        self.chi = chi
        self.d = d
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.sx = torch.Tensor([[0, 1], [1, 0]]).double().to(self.device)
        self.sz = torch.Tensor([[1, 0], [0, -1]]).double().to(self.device)
        self.I = torch.eye(2).double().to(self.device)
        self.clear_data()
        self.data_prev = data_prev

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
        lambdas = self.data_prev["lambdas"] if self.data_prev else lambdas
        for lam in tqdm(lambdas):

            best_model = None
            for _ in range(runs):

                success = False
                while not success:
                    try:
                        model, training_time = self.train_model(
                            lam, epochs, use_prev, perturbation, lr, max_iter
                        )
                        success = True
                    except torch._C._LinAlgError:
                        continue

                if not best_model or model.loss < best_model.loss:
                    best_model = model

            self.collect_data(best_model, training_time, lam)

    def train_model(
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

        A, H, C, T = self.init_tensors(lam, use_prev, perturbation)

        # Map the parameters to a symmetric rank-5 iPEPS tensor
        params, map = torch.unique(A, return_inverse=True)
        params_device = params.to(self.device)
        map_device = map.to(self.device)

        # Initialize the iPEPS model and optimizer
        model = iPEPS(self.chi, self.d, H, params_device, map_device, C, T)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=max_iter, lr=lr)

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _ = model.forward()
            loss.backward()
            return loss

        start_time = time.time()

        for _ in range(epochs):
            loss = optimizer.step(train)

        return model, time.time() - start_time
        # self.collect_data(model, training_time=time.time() - start_time, lam=lam)

    def init_tensors(self, lam: float, use_prev: bool, perturbation: float) -> None:
        H = Tensors.H(lam=lam, sz=self.sz, sx=self.sx, I=self.I)
        C = T = None

        if self.data["states"] and use_prev:
            # Use the last state as the initial state for the next lambda
            A = torch.from_numpy(self.data["states"][-1]).to(self.device)
            C = torch.from_numpy(self.data["C"][-1]).to(self.device)
            T = torch.from_numpy(self.data["T"][-1]).to(self.device)
        elif self.data_prev:
            # Use the last state of the previous data as the initial state
            i = self.data_prev["lambdas"].index(lam)
            A = torch.from_numpy(self.data_prev["states"][i]).to(self.device)
            A = Methods.perturb(A, perturbation)
        else:
            # Initialize a symmetric rank-5 tensor with random values
            A = Tensors.A_random_symmetric(d=self.d).to(self.device)
        return A, H, C, T

    def collect_data(self, model: iPEPS, training_time: float, lam: float) -> None:
        """
        Collect the data from the trained model. The data includes the energy
        and the magnetization components.

        Args:
            model (iPEPS): Trained iPEPS model.
            training_time (float): Time taken to train the model.
        """
        with torch.no_grad():
            E, C, T = model.forward()
            self.data["states"].append(model.params[model.map].detach().cpu().numpy())
            self.data["energies"].append(E)
            self.data["lambdas"].append(lam)
            self.data["train_time"].append(training_time)
            self.data["C"].append(C.detach().cpu().numpy())
            self.data["T"].append(T.detach().cpu().numpy())

    def save_data(self, fn: str = None) -> None:
        """
        Save the collected data to a pickle file. The data is saved in the
        'data' directory.

        Args:
            fn (str): Filename to save the data to. Default is 'data.pkl'.
        """
        directory = "data"
        filename = f"{fn}.pkl" if fn else "data.pkl"

        # Ensure the directory structure exists
        full_path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Save the data to the pickle file
        with open(full_path, "wb") as f:
            pickle.dump(self.data, f)

        print(f"Data saved to {full_path}")

    def clear_data(self) -> None:
        """
        Clear the collected data.
        """
        self.data = {
            "states": [],
            "energies": [],
            "lambdas": [],
            "train_time": [],
            "C": [],
            "T": [],
        }
