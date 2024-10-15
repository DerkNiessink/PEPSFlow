import torch
import os

from pepsflow.models.observables import Observables


class iPEPSReader:
    """
    Class to read iPEPS models from a folder.

    Args:
        folder (str): Folder containing the iPEPS models.
    """

    def __init__(self, folder: str):
        self.folder = folder
        self.filenames = os.listdir(folder)
        self.iPEPS_models = [
            torch.load(os.path.join(folder, fn), weights_only=False)
            for fn in self.filenames
        ]

    def get_lambdas(self) -> list:
        """
        Get the lambda values of the iPEPS models.

        Returns:
            list: List of lambda values.
        """
        return [iPEPS.lam for iPEPS in self.iPEPS_models]

    def get_energies(self) -> list:
        """
        Get the energies of the iPEPS models.

        Returns:
            list: List of energies.
        """
        return [
            iPEPS.forward()[0].detach().cpu().numpy() for iPEPS in self.iPEPS_models
        ]

    def get_magnetizations(self) -> list:
        """
        Get the magnetizations of the iPEPS models.

        Returns:
            list: List of magnetizations.
        """
        magnetizations = []
        for iPEPS in self.iPEPS_models:
            E, C, T = iPEPS.forward()
            A = iPEPS.params[iPEPS.map]
            magnetizations.append(abs(Observables.M(A, C, T)[2].detach().cpu().numpy()))
        return magnetizations

    def get_correlations(self):
        """
        Get the correlations of the iPEPS models.

        Returns:
            list: List of correlations.
        """
        correlations = []
        for iPEPS in self.iPEPS_models:
            E, C, T = iPEPS.forward()
            correlations.append(Observables.xi(T.detach()))
        return correlations

    def get_losses(self, fn: str) -> list:
        """
        Get the losses of the iPEPS model.

        Args:
            fn (str): Filename of the iPEPS model file.

        Returns:
            list: List of losses.
        """
        return torch.load(os.path.join(self.folder, fn), weights_only=False).losses

    def get_iPEPS_state(self, fn: str) -> torch.Tensor:
        """
        Get the iPEPS state from a file.

        Args:
            fn (str): Filename of the iPEPS model file.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters and mapping of the iPEPS model.
        """
        iPEPS = torch.load(os.path.join(self.folder, fn), weights_only=False)
        return iPEPS.params[iPEPS.map]
