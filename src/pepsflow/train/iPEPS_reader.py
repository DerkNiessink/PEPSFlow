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
        iPEPS_models = [
            torch.load(os.path.join(folder, fn), weights_only=False)
            for fn in os.listdir(folder)
        ]
        self.iPEPS_models = {-iPEPS.H[0, 1] * 4: iPEPS for iPEPS in iPEPS_models}

    def get_lambdas(self) -> list:
        """
        Get the lambda values of the iPEPS models.

        Returns:
            list: List of lambda values.
        """
        return list(self.iPEPS_models.keys())

    def get_energies(self) -> list:
        """
        Get the energies of the iPEPS models.

        Returns:
            list: List of energies.
        """
        return [
            iPEPS.forward()[0].detach().cpu().numpy()
            for iPEPS in self.iPEPS_models.values()
        ]

    def get_magnetizations(self) -> list:
        """
        Get the magnetizations of the iPEPS models.

        Returns:
            list: List of magnetizations.
        """
        magnetizations = []
        for iPEPS in self.iPEPS_models.values():
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
        for iPEPS in self.iPEPS_models.values():
            E, C, T = iPEPS.forward()
            correlations.append(Observables.xi(T.detach()))
        return correlations
