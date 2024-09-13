from pepsflow.models.tensors import Tensors
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
        self.Mpx = Tensors.Mpx().to(self.device)
        self.Mpy = Tensors.Mpy().to(self.device)
        self.Mpz = Tensors.Mpz().to(self.device)
        self.sx = torch.Tensor([[0, 1], [1, 0]]).double().to(self.device)
        self.sz = torch.Tensor([[1, 0], [0, -1]]).double().to(self.device)
        self.I = torch.eye(2).double().to(self.device)
        self.inf_tensor = torch.Tensor([float("inf")]).to(self.device)
        self.data = {
            "states": [],
            "energies": [],
            "lambdas": [],
            "Mx": [],
            "My": [],
            "Mz": [],
            "Mg": [],
            "train_time": [],
        }
        self.data_prev = data_prev

    def exe(
        self,
        lambdas: list = [0.5],
        tol: float = 1e-6,
        max_epochs: int = 100,
        use_prev: bool = False,
    ) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.

        Args:
            lambdas (list): List of values of lambda.
        """
        lambdas = self.data_prev["lambdas"] if self.data_prev else lambdas
        for lam in tqdm(lambdas):
            success = False
            while not success:
                try:
                    self.train_model(lam, tol, max_epochs, use_prev)
                    success = True
                except torch._C._LinAlgError:
                    continue

    def train_model(
        self, lam: float, tol: float, max_epochs: int, use_prev: bool
    ) -> None:
        """
        Train the iPEPS model for a given value of lambda.

        Args:
            lam (float): Value of lambda.
            tol (float): Tolerance for the energy convergence.
            max_epochs (int): Maximum number of epochs to train the model.
        """
        H = Tensors.H(lam=lam, sz=self.sz, sx=self.sx, I=self.I)
        if self.data["states"] and use_prev:
            # Use the last state as the initial state for the next lambda
            A = torch.from_numpy(self.data["states"][-1]).to(self.device)
        elif self.data_prev:
            # Use the last state of the previous data as the initial state
            i = self.data_prev["lambdas"].index(lam)
            A = torch.from_numpy(self.data_prev["states"][i]).to(self.device)
        else:
            # Initialize a symmetric rank-5 tensor with random values
            A = Tensors.A_random_symmetric(d=self.d).to(self.device)

        params, map = torch.unique(A, return_inverse=True)
        params_device = params.to(self.device)
        map_device = map.to(self.device)

        model = iPEPS(
            self.chi,
            self.d,
            H,
            self.Mpx,
            self.Mpy,
            self.Mpz,
            params_device,
            map_device,
        )

        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50)

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _, _ = model.forward()
            loss.backward()
            return loss

        start_time = time.time()

        E_new = train()
        diff = torch.abs(E_new - self.inf_tensor)
        epoch = 0
        while diff > tol and epoch < max_epochs:
            loss = optimizer.step(train)
            E_old = E_new
            E_new = loss
            diff = torch.abs(E_new - E_old)
            epoch += 1

        self.collect_data(model, training_time=time.time() - start_time, lam=lam)

    def collect_data(self, model: iPEPS, training_time: float, lam: float) -> None:
        """
        Collect the data from the trained model. The data includes the energy
        and the magnetization components.

        Args:
            model (iPEPS): Trained iPEPS model.
            training_time (float): Time taken to train the model.
        """
        with torch.no_grad():
            E, Mx, My, Mz = model.forward()
            self.data["states"].append(model.params[model.map].detach().cpu().numpy())
            self.data["energies"].append(E)
            self.data["lambdas"].append(lam)
            self.data["Mx"].append(Mx)
            self.data["My"].append(My)
            self.data["Mz"].append(Mz)
            self.data["Mg"].append(torch.sqrt(Mx**2 + My**2 + Mz**2))
            self.data["train_time"].append(training_time)

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
