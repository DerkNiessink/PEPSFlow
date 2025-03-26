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

    def test_general(self):

        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()

        chi_list = [chi for chi in range(2, 20)]
        norm_diff_list = []
        for chi in chi_list:
            alg = CtmGeneral(A, chi=chi)
            alg.exe(N=1)
            norm_diff_list.append(alg.diff4)

        print(norm_diff_list)
        plt.plot(chi_list, norm_diff_list)
        plt.yscale("log")
        plt.xlabel(r"$\chi$")
        plt.ylabel("Norm difference")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.show()
