from pepsflow.models.ctm import CtmSymmetric, CtmGeneral, CtmMirrorSymmetric
from pepsflow.models.tensors import Tensors

import pytest
import torch
import numpy as np


class TestCtmAlg:

    def test_symmmetric(self):
        """
        Test the symmetric CTM algorithm by comparing the evaluated energy of a symmetric Heisenberg state of
        the symmetric CTM algorithms, with and without splitting the bra and ket. The energy should be the same up
        to a small tolerance.
        """
        chi, D = 6, 2

        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)

        A = tensors.A_random_symmetric(D)
        alg_classic = CtmSymmetric(A, chi=chi)
        alg_classic.exe(N=10)
        alg_split = CtmSymmetric(A, chi=chi, split=True)
        alg_split.exe(N=10)

        H = tensors.H_Ising(4)
        rho = tensors.rho_symmetric(A, alg_classic.C, alg_classic.T)
        E = tensors.E(rho, H, which="horizontal")
        rho_split = tensors.rho_symmetric(A, alg_split.C, alg_split.T)
        E_split = tensors.E(rho_split, H, which="horizontal")
        assert E == pytest.approx(E_split, abs=1e-4)

    def test_general(self):
        """
        Test the general CTM algorithm by comparing the evaluated energy of a symmetric Heisenberg state of
        the symmetric and general CTM algorithms. The energy should be the same up to a small tolerance.
        """
        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 24, 2
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        H = tensors.H_Heis_rot()

        alg = CtmGeneral(A, chi=chi)
        alg.exe(N=10)
        alg_symm = CtmSymmetric(A, chi=chi)
        alg_symm.exe(N=10)

        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E_general = (tensors.E(rho, H, which="horizontal") + tensors.E(rho, H, which="vertical")) / 2
        rho_symm = tensors.rho_symmetric(A, alg_symm.C, alg_symm.T)
        E_symm = tensors.E(rho_symm, H, which="horizontal")

        assert E_general == pytest.approx(E_symm, abs=1e-8)

    def test_mirror_symmetric(self):
        """
        Test the mirror symmetric CTM algorithm by comparing the evaluated energy of a symmetric Heisenberg state of
        the symmetric and mirror symmetric CTM algorithms. The energy should be the same up to a small tolerance.
        """
        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 32, 2
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        H = tensors.H_Heis_rot()
        alg = CtmMirrorSymmetric(A, chi=chi, projector_mode="qr")
        alg_symm = CtmSymmetric(A, chi=chi)

        alg.exe(N=50)
        alg_symm.exe(N=50)

        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E_general = (tensors.E(rho, H, which="horizontal") + tensors.E(rho, H, which="vertical")) / 2
        rho_symm = tensors.rho_symmetric(A, alg_symm.C, alg_symm.T)
        E_symm = tensors.E(rho_symm, H, which="horizontal")

        assert E_general == pytest.approx(E_symm, abs=1e-7)

    def test_gauge_change(self):
        """
        Test the gauge invariance of the general CTM algorithm by comparing the evaluated energy of a symmetric
        Heisenberg state before and after a gauge change. The gauge change breaks the symmetry of the state,
        but the energy should remain invariant. The energy should be the same up to a small tolerance.
        """
        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 24, 2
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        H = tensors.H_Heis_rot()
        g1 = tensors.random_unitary(2)
        g2 = tensors.random_unitary(2)
        A_gauge_changed = tensors.gauge_transform(A, g1, g2)

        alg = CtmGeneral(A, chi=chi)
        alg.exe(N=20)

        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E = (tensors.E(rho, H, which="horizontal") + tensors.E(rho, H, which="vertical")) / 2

        alg = CtmGeneral(A_gauge_changed, chi=chi)
        alg.exe(N=20)
        rho = tensors.rho_general(A_gauge_changed, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E_gauge_changed = (tensors.E(rho, H, which="horizontal") + tensors.E(rho, H, which="vertical")) / 2

        assert E_gauge_changed == pytest.approx(E, abs=1e-6)
