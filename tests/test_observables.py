import pytest
import torch
import numpy as np

from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
from pepsflow.models.tensors import Tensors


class TestObservables:

    def test_E_Ising_symmetric(self):
        """
        Test the energy function with the Ising model using the symmetric CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/Ising_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        # Because state is from matlab, we need to permute the dimensions
        A = A.permute(4, 1, 2, 3, 0).contiguous()

        alg = CtmSymmetric(A, chi=16)
        alg.exe(N=100)

        tensors = Tensors(dtype="double", device="cpu")
        assert tensors.E_nn(A, tensors.H_Ising(lam=4), alg.C, alg.T) == pytest.approx(-2.06688, abs=1e-3)

    def test_E_Heisenberg_symmetric(self):
        """
        Test the energy function with the Heisenberg model using the symmetric CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()

        alg = CtmSymmetric(A, chi=48)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu")

        assert tensors.E_nn(A, tensors.H_Heis_rot(), alg.C, alg.T) == pytest.approx(-0.6602310934799582, abs=1e-4)

    def test_E_J1J2_symmetric(self):
        """
        Test the energy function with the J1J2 model using the symmetric CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/J205_D3_state.txt").reshape(2, 3, 3, 3, 3)).double()

        alg = CtmSymmetric(A, chi=16)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu")

        E_nn = tensors.E_nn(A, tensors.H_Heis_rot(), alg.C, alg.T)
        E_nnn = tensors.E_nnn(A, alg.C, alg.T)
        assert E_nn + 0.5 * E_nnn == pytest.approx(-0.49105775959620757, abs=1e-4)

    def test_E_J1J2_general(self):
        """
        Test the energy function with the J1J2 model using the general CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/J205_D3_state.txt").reshape(2, 3, 3, 3, 3)).double()

        alg = CtmGeneral(A, chi=16)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu")

        E_h = tensors.E_horizontal_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_v = tensors.E_vertical_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_nn = (E_h + E_v) / 2

        E_diag_nnn = tensors.E_diagonal_nnn_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E_anti_nnn = tensors.E_anti_diagonal_nnn_general(
            A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_nnn = (E_diag_nnn + E_anti_nnn) / 2

        assert E_nn + 0.5 * E_nnn == pytest.approx(-0.49105775959620757, abs=1e-4)
