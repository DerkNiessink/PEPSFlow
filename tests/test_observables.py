import pytest
import torch
import numpy as np

from pepsflow.models.observables import Observables
from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors


class TestObservables:

    def test_E_Ising(self):
        A = torch.from_numpy(np.loadtxt("tests/Ising_state.txt").reshape(2, 2, 2, 2, 2)).double()

        # Because state is from matlab, we need to permute the dimensions
        A = A.permute(4, 1, 2, 3, 0).contiguous()

        alg = CtmAlg(A, chi=16)
        alg.exe(N=100)

        assert Observables.E(A, Tensors.H_Ising(4), alg.C, alg.T) == pytest.approx(-2.06688, abs=1e-3)

    def test_E_Heisenberg(self):
        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()

        alg = CtmAlg(A, chi=48)
        alg.exe(N=100)

        assert Observables.E(A, Tensors.H_Heisenberg(), alg.C, alg.T) == pytest.approx(-0.6602310934799582, abs=1e-3)
