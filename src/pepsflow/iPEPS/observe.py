import torch

from pepsflow.ipeps.ipeps import iPEPS
from pepsflow.models.tensors import Tensors


class Observer:
    """
    Class to handle the observation of iPEPS models by computing observables
    """

    def __init__(self, ipeps: iPEPS):
        self.ipeps = ipeps
        dtype = self.ipeps.args.get("dtype", "double")
        device = self.ipeps.args.get("device", "cpu")
        self.tensors = Tensors(dtype=dtype, device=device)

    def lam(self) -> float:
        """Get the lambda value (field strength of Ising model)."""
        return self.ipeps.args["lam"]

    def losses(self) -> list[float]:
        """Get the losses."""
        return self.ipeps.data["energies"]

    def gradient_norms(self) -> list[float]:
        """Get the gradient norms."""
        return [norm.detach().cpu() for norm in self.ipeps.data["norms"]]

    def state(self) -> torch.Tensor:
        """Get the iPEPS state from the iPEPS model. These values are NOT mapped to their original positions."""
        return self.ipeps.params.detach().cpu()

    def energy(self) -> float:
        """Get the energy."""
        return self.ipeps.data["energies"][-1]

    def magnetization(self) -> float:
        """Get the magnetization."""
        A = self.ipeps.params[self.ipeps.map]
        C, T = self.ipeps.do_evaluation()
        return float(abs(self.tensors.M(A, C, T)[2].cpu()))

    def correlation(self) -> float:
        """Get the correlation."""
        C, T = self.ipeps.do_evaluation()
        return float(self.tensors.xi(T).cpu())

    def ctm_steps(self) -> list:
        """Get the number of CTM steps."""
        return self.ipeps.data["Niter_warmup"]

    def warmup_steps(self) -> list:
        """Get the number of warmup steps."""
        return self.ipeps.data["warmup_steps"]

    def chi(self) -> int:
        """Get the bond dimension."""
        return self.ipeps.args["chi"]

    def Niter(self) -> int:
        """Get the number of iterations."""
        return self.ipeps.args["Niter"]

    def ipeps_args(self) -> dict:
        """Get the iPEPS arguments."""
        return self.ipeps.args

    def eval_energy(self) -> float:
        """Get the evaluation energy."""
        return self.ipeps.data["Eval_energy"][-1] if "Eval_energy" in self.ipeps.data else None
