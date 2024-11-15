from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors
from pepsflow.models.observables import Observables

import pytest
import torch


class TestCtmAlg:

    def test_exe(self):

        A = Tensors.A_random_symmetric(D=2)
        alg_classic = CtmAlg(A, chi=6)
        alg_classic.exe(N=10)
        alg_split = CtmAlg(A, chi=6, split=True)
        alg_split.exe(N=10)

        H = Tensors.H_Ising(4)
        E = Observables.E(A, H, alg_classic.C, alg_classic.T)
        E_split = Observables.E(A, H, alg_split.C, alg_split.T)

        assert E == pytest.approx(E_split, abs=1e-4)
