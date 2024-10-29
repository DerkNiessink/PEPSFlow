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

        alg = CtmAlg(a=Tensors.a(A), chi=16)
        alg.exe(max_steps=100)

        sx = torch.Tensor([[0, 1], [1, 0]]).double()
        sz = torch.Tensor([[1, 0], [0, -1]]).double()
        I = torch.eye(2).double()
        H = Tensors.H_Ising(lam=4, sz=sz, sx=sx, I=I)

        assert Observables.E(A, H, alg.C, alg.T) == pytest.approx(-2.06688, abs=1e-3)

    def test_E_Heisenberg(self):
        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()

        alg = CtmAlg(a=Tensors.a(A), chi=48)
        alg.exe(max_steps=100)

        sz = torch.Tensor([[1, 0], [0, -1]]).double()
        sy = torch.Tensor([[0, -1], [1, 0]]).double()
        sy = torch.complex(torch.zeros_like(sy), sy)
        sp = torch.Tensor([[0, 1], [0, 0]]).double()
        sm = torch.Tensor([[0, 0], [1, 0]]).double()
        H = Tensors.H_Heisenberg(lam=1, sz=sz, sp=sp, sm=sm, sy=sy)
        assert Observables.E(A, H, alg.C, alg.T) == pytest.approx(-0.6602310934799582, abs=1e-3)
