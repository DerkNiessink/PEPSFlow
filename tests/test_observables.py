import pytest
import torch
import numpy as np

from pepsflow.models.observables import Observables
from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors


class TestObservables:

    def test_E(self):
        A = torch.from_numpy(
            np.loadtxt("tests/solution_state.txt").reshape(2, 2, 2, 2, 2)
        ).double()
        A = A.permute(4, 1, 2, 3, 0).contiguous()

        alg = CtmAlg(a=Tensors.a(A), chi=16)
        alg.exe()

        sx = torch.Tensor([[0, 1], [1, 0]]).double()
        sz = torch.Tensor([[1, 0], [0, -1]]).double()
        I = torch.eye(2).double()
        H = Tensors.H_Ising(lam=4, sz=sz, sx=sx, I=I)

        assert Observables.E(A, H, alg.C, alg.T) == pytest.approx(-2.06688, abs=1e-3)
