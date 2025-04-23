from pepsflow.models.ctm import CtmSymmetric, CtmGeneral
from pepsflow.models.tensors import Tensors

import pytest
import torch
import numpy as np


class TestProjectorModes:
    """
    Test the projector modes of the iPEPS class.
    """

    @pytest.mark.parametrize("projector_mode", ["eig", "svd", "iterative_eig", "qr"])
    def test_projector_modes_symmetric(self, projector_mode):
        """
        Test the projector modes for the symmetric CTM algorithm.
        """
        A = torch.from_numpy(np.loadtxt("tests/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        tensors = Tensors(dtype="double", device="cpu")

        alg = CtmSymmetric(A, chi=6, projector_mode=projector_mode)
        alg.exe(N=10)
        E = tensors.E_nn(A, tensors.H_Heis_rot(), alg.C, alg.T)

        # Check that the projector modes are correct
        assert E == pytest.approx(-0.6602310934799582, abs=1e-4)  # Replace with actual expected values

    @pytest.mark.skip()
    def test_projector_modes_general(self):
        """
        Test the projector modes for the general CTM algorithm.
        """
        A = torch.from_numpy(np.loadtxt("tests/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        tensors = Tensors(dtype="double", device="cpu")

        alg = CtmGeneral(A, chi=300, projector_mode="qr")
        alg.exe(N=2)
        E_h = tensors.E_vertical_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_v = tensors.E_horizontal_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_nn = (E_h + E_v) / 2

        # Check that the projector modes are correct
        assert E_nn == pytest.approx(-0.6602310934799582, abs=1e-5)
