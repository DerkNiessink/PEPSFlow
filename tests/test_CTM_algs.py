from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
from pepsflow.models.tensors import Tensors

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt


class TestCtmAlg:

    def test_symmmetric(self):

        tensors = Tensors(dtype="double", device="cpu")

        A = tensors.A_random_symmetric(D=2)
        alg_classic = CtmSymmetric(A, chi=6)
        alg_classic.exe(N=10)
        alg_split = CtmSymmetric(A, chi=6, split=True)
        alg_split.exe(N=10)

        H = tensors.H_Ising(4)
        E = tensors.E_nn(A, H, alg_classic.C, alg_classic.T)
        E_split = tensors.E_nn(A, H, alg_split.C, alg_split.T)

        assert E == pytest.approx(E_split, abs=1e-4)

    def test_general_Heis(self):

        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()
        alg = CtmGeneral(A, chi=6)
        alg.exe(N=100)
        alg_symm = CtmSymmetric(A, chi=6)
        alg_symm.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu")
        E_general = tensors.E_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_symm = tensors.E_nn(A, tensors.H_Heis_rot(), alg_symm.C, alg_symm.T)

        assert E_general == pytest.approx(E_symm, abs=1e-4)

    def test_general_Ising(self):
        A = torch.from_numpy(np.loadtxt("tests/Ising_state.txt").reshape(2, 2, 2, 2, 2)).double()
        alg = CtmGeneral(A, chi=6)
        alg.exe(N=100)
        alg_symm = CtmSymmetric(A, chi=6)
        alg_symm.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu")
        E_general = tensors.E_nn_general(
            A, tensors.H_Ising(lam=4), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_symm = tensors.E_nn(A, tensors.H_Ising(lam=4), alg_symm.C, alg_symm.T)

        assert E_general == pytest.approx(E_symm, abs=1e-4)
