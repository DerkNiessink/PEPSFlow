from pepsflow.models.ctm import CtmSymmetric, CtmGeneral, CtmMirrorSymmetric
from pepsflow.models.tensors import Tensors
from pepsflow.models.canonize import apply_minimal_canonical, apply_simple_update, minimal_canonical_criterion

import torch.utils.checkpoint as checkpoint
from abc import ABC, abstractmethod
from contextlib import contextmanager
import torch
import numpy as np
from typing import Any


def make_ipeps(args: dict, initial_ipeps: "iPEPS" = None) -> "iPEPS":
    """
    iPEPS is a class of tensor networks used to represent a quantum many-body state and perform
    variational optimization.

    Args:
        args (dict): Dictionary containing the iPEPS parameters.
        initial_ipeps (iPEPS, optional): Initial iPEPS object to use as a starting point.

    Returns:
        iPEPS: An instance of the iPEPS class with the specified symmetry.
    """
    match args["ctm_symmetry"]:
        case "rotational":
            return RotationalSymmetricIPEPS(args, initial_ipeps)
        case "mirror":
            return MirrorSymmetricIPEPS(args, initial_ipeps)
        case None:
            return GeneralIPEPS(args, initial_ipeps)
        case _:
            raise ValueError(f"Unknown symmetry: {args['ctm_symmetry']}, expected 'rotational', 'mirror', or None.")


class iPEPS(torch.nn.Module, ABC):
    def __init__(self, args: dict, initial_ipeps: "iPEPS" = None):
        super().__init__()
        self.args = args
        self.set_seed(args["seed"])
        self.initial_ipeps = initial_ipeps
        self.tensors = Tensors(args["dtype"], args["device"], self.args["chi"], args["D"])
        self.data = {}
        self.to(args["device"])
        self.H = self.tensors.Hamiltonian(args["model"], lam=args["lam"])
        self.params, self.map = None, None
        self._setup_random() if initial_ipeps is None else self._setup_from_initial_ipeps()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)

    def add_data(self, key: str, value: Any):
        """Add data to the iPEPS object. If the key already exists, appends the value to the list."""
        if self.data.get(key) is None:
            self.data[key] = []
        self.data.setdefault(key, []).append(value)

    def set_data(self, key: str, value: Any):
        """Set data in the iPEPS object. If the key already exists, overwrites the value."""
        self.data[key] = value

    def do_warmup_steps(self, N: int) -> tuple[torch.Tensor, ...]:
        """Warmup the iPEPS tensor by performing the CTM algorithm without gradient tracking.

        Args:
            N (int): Number of iterations in the CTM algorithm.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        return self._forward(N, grad=False)

    def do_gradient_steps(self, N: int, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Take gradient steps in the optimization of the iPEPS tensor.

        args:
            N (int): Number of iterations in the CTM algorithm.
            tensors (tuple): Tuple containing initial tensors for the CTM algorithm, obtained from the warmup steps.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        if self.args["use_checkpoint"]:
            return checkpoint.checkpoint(self._forward, N, grad=True, tensors=tensors, use_reentrant=False)
        else:
            return self._forward(N, grad=True, tensors=tensors)

    def do_evaluation(self, N: int, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Evaluate the iPEPS tensor by performing the CTM algorithm without gradient tracking.

        args:
            N (int): Number of iterations in the CTM algorithm.
            kwargs (dict): Additional arguments to override the iPEPS parameters, such as chi, ctm_symmetry,
            and projector_mode.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        # Temporarily override chi and ctm_symmetry for evaluation
        with self._temporary_args_override(**kwargs):
            tensors = self._forward(N, grad=False)
        return tensors

    @contextmanager
    def _temporary_args_override(self, **overrides):
        """
        Context manager to temporarily override arguments in self.args.

        Args:
            **overrides: Key-value pairs of arguments to temporarily override.

        Yields:
            None
        """
        original_args = {key: self.args[key] for key in overrides}
        self.args.update(overrides)
        try:
            yield
        finally:
            self.args.update(original_args)

    @abstractmethod
    def _setup_random(self):
        """Setup the iPEPS tensor with a random initial state."""
        pass

    @abstractmethod
    def _setup_from_initial_ipeps(self):
        """Setup the iPEPS tensor from an initial state."""
        pass

    @abstractmethod
    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        """
        Perform a forward pass of the iPEPS tensor network using the CTM algorithm.

        Args:
            N (int): Number of iterations in the CTM algorithm.
            grad (bool): Whether to compute gradients.
            tensors (tuple): Tuple containing the tensors needed to perform the CTM algorithm.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing the corner and edge tensors of the iPEPS tensor network.
        """
        pass

    @abstractmethod
    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute and set the energy of the iPEPS tensor network.

        args:
            grad (bool): Whether to compute gradients.
            tensors (tuple): Tuple containing the tensors needed to perform the CTM algorithm.

        Returns:
            torch.Tensor: Energy of the iPEPS tensor network.
        """
        pass


class RotationalSymmetricIPEPS(iPEPS):
    """
    Class representing a rotationally symmetric iPEPS state. This class uses a mapping to
    reduce the number of parameters in the iPEPS tensor network. This iPEPS subclass uses the
    rotational symmetric ctm algorithm, which uses only one corner and one edge tensor.
    """

    def _setup_random(self):
        A = self.tensors.A_random_symmetric(self.args["D"])
        params, self.map = torch.unique(A, return_inverse=True)
        self.params = torch.nn.Parameter(params)

    def _setup_from_initial_ipeps(self):
        self.data = self.initial_ipeps.data
        params = self.initial_ipeps.params.detach()
        self.params = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"])
        self.map = self.initial_ipeps.map

    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        A = self.params[self.map] if self.map is not None else self.params
        A = A.detach() if not grad else A
        C, T = tensors
        rho = self.tensors.rho_symmetric(A, C, T)
        E_nn = self.tensors.E(rho, self.H, which="horizontal")
        if self.args["model"] == "J1J2":
            return E_nn + self.args["J2"] * self.tensors.E(rho, self.tensors.H_Heis(), which="diagonal")
        return E_nn

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        A = self.params[self.map]
        A = A.detach() if not grad else A
        alg = CtmSymmetric(A, self.args["chi"], tensors, self.args["split"], self.args["projector_mode"])
        alg.exe(N)
        return alg.C, alg.T


class GeneralIPEPS(iPEPS):
    """
    Class representing a general iPEPS state. This subclasss uses the general ctm algorithm,
    which uses four corner and four edge tensors. It also implements an additional method to
    gauge transform the iPEPS state.
    """

    def _setup_random(self):
        match self.args["initial_state_symmetry"]:
            case "rotational":
                A = self.tensors.A_random_symmetric(self.args["D"])
            case "mirror":
                # TODO: Implement mirror symmetry for the initial state
                A = self.tensors.A_random_symmetric(self.args["D"])
            case None:
                A = self.tensors.A_random(self.args["D"])
            case _:
                raise ValueError(
                    f"Unknown initial state symmetry: {self.args['initial_state_symmetry']}, expected 'rotational', 'mirror', or 'general'."
                )
        self.params = torch.nn.Parameter(A / A.norm())

    def _setup_from_initial_ipeps(self):
        self.data = self.initial_ipeps.data
        params = self.initial_ipeps.params.detach()
        if self.initial_ipeps.map is not None:
            # Case where the initial iPEPS is a symmetric state
            params = params[self.initial_ipeps.map]
        self.params = torch.nn.Parameter(params + torch.randn_like(params) * self.args["noise"])

    def get_E(self, grad: bool, tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
        A = self.params.detach() if not grad else self.params

        C, T = tensors[:4], tensors[4:]
        rho = self.tensors.rho_general(A, *C, *T)
        Eh = self.tensors.E(rho, self.H, which="horizontal")
        Ev = self.tensors.E(rho, self.H, which="vertical")
        E_nn = (Eh + Ev) / 2

        if self.args["model"] == "J1J2":
            Ed = self.tensors.E(rho, self.tensors.H_Heis(), which="diagonal")
            Ead = self.tensors.E(rho, self.tensors.H_Heis(), which="antidiagonal")
            E_nnn = (Ed + Ead) / 2
            return E_nn + self.args["J2"] * E_nnn

        return E_nn

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        A = self.params.detach() if not grad else self.params

        if self.args["ctm_symmetry"] == "mirror":
            alg = CtmMirrorSymmetric(A, self.args["chi"], tensors, projector_mode=self.args["projector_mode"])
        else:
            alg = CtmGeneral(A, self.args["chi"], tensors)

        alg.exe(N)
        return alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4

    def gauge_transform(self, which: str, tolerance: float = 1e-16) -> None:
        """Apply a gauge transformation to the iPEPS tensor.

        Args:
            which (str): Type of gauge transformation to apply. Options are "minimal_canonical",
            "simple_update", "invertible" and "unitary". If None is given, the gauge transformation
            is an identity.
            Tolerance (float): Tolerance for the "minimal_canonical" gauge transformation. Default is 1e-16.
        """
        if which == "minimal_canonical":
            A = apply_minimal_canonical(self.params, tolerance)
        elif which == "simple_update":
            A = apply_simple_update(self.params, tolerance)
        else:
            g1, g2 = self.tensors.gauges(self.args["D"], which=which)
            A = self.tensors.gauge_transform(self.params, g1, g2)
        with torch.no_grad():
            self.params.data.copy_(A)

    def minimal_canonical_criterion(self) -> float:
        """Compute the norm of the iPEPS state."""
        A = self.params.detach()
        return minimal_canonical_criterion(A).item()


class MirrorSymmetricIPEPS(GeneralIPEPS):
    """
    Class representing a mirror symmetric iPEPS state. Becaus the mirror symmetric ctm also uses
    four corner and four edge tensors, this class inherits from the GeneralIPEPS class and is almost
    identical to it.
    """

    def _forward(self, N: int, grad: bool, tensors: tuple[torch.Tensor, ...] = None) -> tuple:
        A = self.params.detach() if not grad else self.params
        alg = CtmMirrorSymmetric(A, self.args["chi"], tensors, projector_mode=self.args["projector_mode"])
        alg.exe(N)
        return alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
