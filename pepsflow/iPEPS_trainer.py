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
    """

    def __init__(self, chi: int, d: int, gpu: bool = False):
        self.chi = chi
        self.d = d
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.Mpx = Tensors.Mpx().to(self.device)
        self.Mpy = Tensors.Mpy().to(self.device)
        self.Mpz = Tensors.Mpz().to(self.device)
        self.inf_tensor = torch.Tensor([float("inf")]).to(self.device)
        self.data = {
            "states": [],
            "energies": [],
            "Mx": [],
            "My": [],
            "Mz": [],
            "Mg": [],
        }

    def exe(
        self,
        lambdas: list,
        tol: float = 1e-6,
        max_epochs: int = 100,
        use_prev: bool = False,
    ) -> None:
        """
        Execute the training of the iPEPS model for different values of lambda.

        Args:
            lambdas (list): List of values of lambda.
        """
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
        H = Tensors.H(lam=lam).to(self.device)
        # Use the last state as the initial state for the next lambda
        A = (
            torch.from_numpy(self.data["states"][-1]).to(self.device)
            if self.data["states"] and use_prev
            else None
        )
        model = iPEPS(
            chi=self.chi, d=self.d, H=H, Mpx=self.Mpx, Mpy=self.Mpy, Mpz=self.Mpz, A=A
        ).to(self.device)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50)

        def train() -> torch.Tensor:
            """
            Train the parameters of the iPEPS model.
            """
            optimizer.zero_grad()
            loss, _, _, _ = model.forward()
            loss.backward()
            return loss

        E_new = train()
        diff = torch.abs(E_new - self.inf_tensor)
        epoch = 0
        while diff > tol and epoch < max_epochs:
            loss = optimizer.step(train)
            E_old = E_new
            E_new = loss
            diff = torch.abs(E_new - E_old)
            epoch += 1

        self.collect_data(model)

    def collect_data(self, model: iPEPS) -> None:
        """
        Collect the data from the trained model. The data includes the energy
        and the magnetization components.

        Args:
            model (iPEPS): Trained iPEPS model.
        """
        with torch.no_grad():
            E, Mx, My, Mz = model.forward()
            self.data["states"].append(model.A.detach().cpu().numpy())
            self.data["energies"].append(E)
            self.data["Mx"].append(Mx)
            self.data["My"].append(My)
            self.data["Mz"].append(Mz)
            self.data["Mg"].append(torch.sqrt(Mx**2 + My**2 + Mz**2))

    def save_data(self, fn: str = None) -> None:
        """
        Save the collected data to a pickle file. The data is saved in the
        'data' directory. If the filename already exists, a new filename is
        created with a counter appended to the original filename.

        Args:
            fn (str): Filename to save the data to. Default is 'data.pkl'.
        """
        directory = "data"
        filename = f"{fn}.pkl" if fn else "data.pkl"

        # Ensure the directory structure exists
        full_path = os.path.join(directory, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Check if file exists, and modify filename if necessary
        if os.path.exists(full_path):
            base, ext = os.path.splitext(full_path)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}"):
                counter += 1
            full_path = f"{base}_{counter}{ext}"

        # Save the data to the pickle file
        with open(full_path, "wb") as f:
            pickle.dump(self.data, f)

        print(f"Data saved to {full_path}")
