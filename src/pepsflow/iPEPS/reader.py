import torch

from pepsflow.models.tensors import Tensors
from pepsflow.iPEPS.iPEPS import iPEPSBase


class Reader:
    """
    Class to read an iPEPS model from a file.

    Args:
        file (str): File containing the iPEPS model.
    """

    def __init__(self, file: str):
        self.file = f"{file}.pth" if not file.endswith(".pth") else file
        self.ipeps: iPEPSBase = torch.load(self.file, weights_only=False)
        dtype = self.ipeps.args.get("dtype", "double")
        device = self.ipeps.args.get("device", "cpu")
        self.tensors = Tensors(dtype, device)

    def lam(self) -> float:
        """
        Get the lambda value of the iPEPS model.

        Returns:
            float: Lambda value.
        """
        return self.ipeps.args["lam"]

    def losses(self) -> list[float]:
        """
        Get the losses of the iPEPS model.

        Returns:
            list: List of losses.
        """
        return self.ipeps.data["losses"]

    def gradient_norms(self) -> list[float]:
        """
        Get the gradient norms of the iPEPS model.

        Returns:
            list: List of gradient norms.
        """
        return [norm.detach().cpu() for norm in self.ipeps.data["norms"]]

    def iPEPS_state(self) -> torch.Tensor:
        """
        Get the iPEPS state from the iPEPS model. These values are NOT mapped to their
        original positions.

        Returns:
            torch.Tensor: iPEPS state
        """
        return self.ipeps.params.detach().cpu()

    def energy(self) -> float:
        """
        Get the energy of the iPEPS model.

        Returns:
            float: Energy of the iPEPS model.
        """
        return self.ipeps.data["losses"][-1]

    def magnetization(self) -> float:
        """
        Get the magnetization of the iPEPS model.

        Returns:
            float: Magnetization of the iPEPS model.
        """
        A = self.ipeps.params[self.ipeps.map]
        C, T = self.ipeps.do_evaluation()
        return float(abs(self.tensors.M(A, C, T)[2].cpu()))

    def correlation(self) -> float:
        """
        Get the correlation of the iPEPS model.

        Returns:
            float: Correlation of the iPEPS model.
        """
        C, T = self.ipeps.do_evaluation()
        return float(self.tensors.xi(T).cpu())

    def ctm_steps(self) -> list:
        """
        Get the number of CTM steps before convergence of each epoch for the iPEPS model

        Returns:
            list: List of number of CTM steps.
        """
        return self.ipeps.data["Niter_warmup"]
