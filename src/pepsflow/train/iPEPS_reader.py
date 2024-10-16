import torch
import os

from pepsflow.models.observables import Observables


class iPEPSReader:
    """
    Class to read an iPEPS model from a file.

    Args:
        file (str): File containing the iPEPS model.
    """

    def __init__(self, file: str):
        self.iPEPS = torch.load(file, weights_only=False)

    def get_lam(self) -> float:
        """
        Get the lambda value of the iPEPS model.

        Returns:
            float: Lambda value.
        """
        return self.iPEPS.lam

    def get_energy(self) -> float:
        """
        Get the energy of the iPEPS model.

        Returns:
            float: Energy of the iPEPS model.
        """
        return float(self.iPEPS.forward()[0].detach().cpu().numpy())

    def get_magnetization(self) -> float:
        """
        Get the magnetization of the iPEPS model.

        Returns:
            float: Magnetization of the iPEPS model.
        """
        E, C, T = self.iPEPS.forward()
        A = self.iPEPS.params[self.iPEPS.map]
        return float(abs(Observables.M(A, C, T)[2].detach().cpu().numpy()))

    def get_correlation(self) -> float:
        """
        Get the correlation of the iPEPS model.

        Returns:
            float: Correlation of the iPEPS model.
        """
        E, C, T = self.iPEPS.forward()
        return float(Observables.xi(T.detach()).cpu().numpy())

    def get_losses(self) -> list[float]:
        """
        Get the losses of the iPEPS model.

        Returns:
            list: List of losses.
        """
        return self.iPEPS.losses

    def get_iPEPS_state(self) -> torch.Tensor:
        """
        Get the iPEPS state from the iPEPS model.

        Returns:
            torch.Tensor: iPEPS state
        """
        return self.iPEPS.params[self.iPEPS.map]
