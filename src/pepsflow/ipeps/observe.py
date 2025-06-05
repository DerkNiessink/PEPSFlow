import torch

from pepsflow.ipeps.ipeps import iPEPS
from pepsflow.models.tensors import Tensors


class Observer:
    """Class to handle the observation of iPEPS models"""

    def __init__(self, ipeps: iPEPS):
        self.ipeps = ipeps
        dtype = self.ipeps.args.get("dtype", "double")
        device = self.ipeps.args.get("device", "cpu")
        self.tensors = Tensors(dtype=dtype, device=device)

    def lam(self) -> float:
        """Get the lambda value (field strength of Ising model)."""
        return self.ipeps.args["lam"]

    def optimization_energies(self, i: int = None) -> list[float]:
        """
        Get the optimization energies.

        Args:
            i (int): The index of the optimization data to retrieve. If None, return all energies.
        """
        if i is None:
            energies = []
            for opt_data in self.ipeps.data["optimization"]:
                energies.extend(opt_data["energies"])
        else:
            energies = self.ipeps.data["optimization"][i]["energies"]
        return energies

    def optimization_norms(self, i: int = None) -> list[float]:
        """
        Get the optimization norms.

        Args:
            i (int): The index of the optimization data to retrieve. If None, return all norms.
        """
        if i is None:
            norms = []
            for opt_data in self.ipeps.data["optimization"]:
                norms.extend(opt_data["norms"])
        else:
            norms = self.ipeps.data["optimization"][i]["norms"]
        return norms

    def optimization_energy(self) -> float:
        """Get the last optimization energy."""
        return self.ipeps.data["optimization"][-1]["energies"][-1]

    def optimization_warmup_ctm_steps(self, i: int) -> int:
        """
        Get the number of CTM warmup steps.

        Args:
            i (int): The index of the optimization data to retrieve.
        """
        return self.ipeps.data["optimization"][i]["args"]["warmup_steps"]

    def optimization_gradient_ctm_steps(self, i: int) -> list:
        """
        Get the number of CTM gradient steps.

        Args:
            i (int): The index of the optimization data to retrieve.
        """
        return self.ipeps.data["optimization"][i]["args"]["gradient_steps"]

    def optimization_chi(self) -> int:
        """Get the optimization bond dimension."""
        return self.ipeps.args["chi"]

    def optimization_args(self, i: int) -> dict:
        """
        Get the optimization arguments.

        Args:
            i (int): The index of the optimization data to retrieve.
        """
        return self.ipeps.data["optimization"][i]["args"]

    def ipeps_args(self) -> dict:
        """Get the iPEPS arguments."""
        return self.ipeps.args

    def state(self) -> torch.Tensor:
        """Get the iPEPS state from the iPEPS model. These values are NOT mapped to their original positions."""
        return self.ipeps.params.detach().cpu()

    # TODO: Fix this function
    def magnetization(self) -> float:
        """Get the magnetization."""
        A = self.ipeps.params[self.ipeps.map]
        C, T = self.ipeps.do_evaluation(N=20, chi=32, ctm_symmetry="rotational", projector_mode="qr")
        return float(abs(self.tensors.M(A, C, T)[2].cpu()))

    # TODO: Fix this function
    def correlation(self) -> float:
        """Get the correlation."""
        C, T = self.ipeps.do_evaluation(N=20, chi=32, ctm_symmetry="rotational", projector_mode="qr")
        return float(self.tensors.xi(T).cpu())

    def evaluation_data(self) -> list[dict]:
        """Get all evaluation data"""
        return self.ipeps.data["evaluation"]

    def evaluation_energies(self, i: int) -> float | None:
        """
        Get the evaluation energy.

        Args:
            i (int): The index of the evaluation data to retrieve.
        """
        return self.ipeps.data["evaluation"][i]["energies"]

    def evaluation_chis(self, i: int) -> int | None:
        """
        Get the evaluation chis.

        Args:
            i (int): The index of the evaluation data to retrieve.
        """
        return self.ipeps.data["evaluation"][i]["args"]["chi"]
