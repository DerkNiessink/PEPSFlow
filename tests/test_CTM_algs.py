from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
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
        """
        Test the general CTM algorithm by comparing the evaluated energy of a symmetric Heisenberg state of
        the symmetric and general CTM algorithms. The energy should be the same up to a small tolerance.
        """
        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()
        alg = CtmGeneral(A, chi=24)
        alg.exe(N=10)
        alg_symm = CtmSymmetric(A, chi=24)
        alg_symm.exe(N=10)
        tensors = Tensors(dtype="double", device="cpu")
        E_general = tensors.E_vertical_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_general += tensors.E_horizontal_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_symm = tensors.E_nn(A, tensors.H_Heis_rot(), alg_symm.C, alg_symm.T)
        assert E_general / 2 == pytest.approx(E_symm, abs=1e-8)

    def test_gauge_change(self):
        """
        Test the gauge invariance of the general CTM algorithm by comparing the evaluated energy of a symmetric
        Heisenberg state before and after a gauge change. The gauge change breaks the symmetry of the state,
        but the energy should remain invariant. The energy should be the same up to a small tolerance.
        """
        A = torch.from_numpy(np.loadtxt("tests/Heisenberg_state.txt").reshape(2, 2, 2, 2, 2)).double()
        tensors = Tensors(dtype="double", device="cpu")
        U1 = tensors.random_unitary(2)
        U2 = tensors.random_unitary(2)
        A_gauge_changed = torch.einsum("abcde,bf,cg,dh,ei->afghi", A, U1, U2, U1, U2)

        alg = CtmGeneral(A, chi=24)
        alg.exe(N=10)
        E = tensors.E_horizontal_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E += tensors.E_vertical_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )

        alg = CtmGeneral(A_gauge_changed, chi=24)
        alg.exe(N=10)
        E_gauge_changed = tensors.E_horizontal_nn_general(
            A_gauge_changed, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_gauge_changed += tensors.E_vertical_nn_general(
            A_gauge_changed, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )

        assert E_gauge_changed == pytest.approx(E, abs=1e-6)
