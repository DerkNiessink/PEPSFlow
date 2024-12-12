import torch

from pepsflow.models.tensors import Tensors
from pepsflow.iPEPS.iPEPS import iPEPS


class iPEPSReader:
    """
    Class to read an iPEPS model from a file.

    Args:
        file (str): File containing the iPEPS model.
    """

    def __init__(self, file: str):
        self.file = f"{file}.pth" if not file.endswith(".pth") else file
        self.iPEPS: iPEPS = torch.load(self.file, weights_only=False)
        self.iPEPS.eval()
        self.tensors = Tensors(self.iPEPS.args["dtype"], self.iPEPS.args["device"])

    def lam(self) -> float:
        """
        Get the lambda value of the iPEPS model.

        Returns:
            float: Lambda value.
        """
        return self.iPEPS.args["lam"]

    def losses(self) -> list[float]:
        """
        Get the losses of the iPEPS model.

        Returns:
            list: List of losses.
        """
        return [E.detach() for E in self.iPEPS.data["losses"]]

    def gradient_norms(self) -> list[float]:
        """
        Get the gradient norms of the iPEPS model.

        Returns:
            list: List of gradient norms.
        """
        return self.iPEPS.data["norms"]

    def iPEPS_state(self) -> torch.Tensor:
        """
        Get the iPEPS state from the iPEPS model.

        Returns:
            torch.Tensor: iPEPS state
        """
        return self.iPEPS.params.detach()

    def energy(self) -> float:
        """
        Get the energy of the iPEPS model.

        Returns:
            float: Energy of the iPEPS model.
        """
        return float(self.iPEPS.data["losses"][-1].detach())

    def magnetization(self) -> float:
        """
        Get the magnetization of the iPEPS model.

        Returns:
            float: Magnetization of the iPEPS model.
        """
        A = self.iPEPS.params[self.iPEPS.map]
        return float(abs(self.tensors.M(A, self.iPEPS.C, self.iPEPS.T)[2]))

    def correlation(self) -> float:
        """
        Get the correlation of the iPEPS model.

        Returns:
            float: Correlation of the iPEPS model.
        """
        return float(self.tensors.xi(self.iPEPS.T))

    def set_to_lowest_energy(self) -> None:
        """
        Set the iPEPS model to the state with the lowest energy.
        """
        self.iPEPS.set_to_lowest_energy()
        torch.save(self.iPEPS, self.file)
