from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.CTM_alg_split import CtmAlgSplit
from pepsflow.models.tensors import Tensors
from pepsflow.models.observables import Observables

import pytest
import torch


@pytest.mark.skip(reason="Skipping this test for now")
class TestCtmAlg:

    def test_exe(self):
        A = Tensors.A_random_symmetric(D=2)
        alg = CtmAlg(a=Tensors.a(A), chi=2)
        alg.exe()
        alg_split = CtmAlgSplit(A=A, chi=2)
        alg_split.exe()

        sx = torch.Tensor([[0, 1], [1, 0]]).double()
        sz = torch.Tensor([[1, 0], [0, -1]]).double()
        I = torch.eye(2).double()
        H = Tensors.H_Ising(lam=4, sz=sz, sx=sx, I=I)

        E = Observables.E(A, H, alg.C, alg.T)
        E_split = Observables.E(A, H, alg_split.C, alg_split.T)

        assert E == pytest.approx(E_split, abs=1e-3)
