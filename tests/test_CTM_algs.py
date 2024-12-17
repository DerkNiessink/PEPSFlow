from pepsflow.models.CTM_alg import CtmAlg
from pepsflow.models.tensors import Tensors

import pytest


class TestCtmAlg:

    def test_exe(self):

        tensors = Tensors(dtype="double", device="cpu")

        A = tensors.A_random_symmetric(D=2)
        alg_classic = CtmAlg(A, chi=6)
        alg_classic.exe(N=10)
        alg_split = CtmAlg(A, chi=6, split=True)
        alg_split.exe(N=10)

        H = tensors.H_Ising(4)
        E = tensors.E_nn(A, H, alg_classic.C, alg_classic.T)
        E_split = tensors.E_nn(A, H, alg_split.C, alg_split.T)

        assert E == pytest.approx(E_split, abs=1e-4)
