from pepsflow.models.ctm import CtmSymmetric, CtmMirrorSymmetric
from pepsflow.models.tensors import Tensors

import pytest
import torch
import numpy as np


class TestProjectorModes:
    """
    Test the projector modes of the iPEPS class.
    """

    @pytest.mark.parametrize("projector_mode", ["eig", "svd", "iterative_eig", "qr"])
    def test_projector_modes_rotational_symmetric(self, projector_mode):
        """
        Test the projector modes for the symmetric CTM algorithm.
        """
        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 6, 2
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)

        alg = CtmSymmetric(A, chi=chi, projector_mode=projector_mode)
        alg.exe(N=10)
        rho = tensors.rho_symmetric(A, alg.C, alg.T)
        E = tensors.E(rho, tensors.H_Heis_rot(), which="horizontal")

        # Check that the projector modes are correct
        assert E == pytest.approx(-0.6602310934799582, abs=1e-8)

    @pytest.mark.parametrize("projector_mode", ["svd", "qr", "improved_qr"])
    def test_projector_modes_mirror_symmetric(self, projector_mode):
        """
        Test the projector modes for the symmetric CTM algorithm.
        """
        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 24, 2
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        alg = CtmMirrorSymmetric(A, chi=chi, projector_mode=projector_mode)
        alg.exe(N=10)
        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E = (
            tensors.E(rho, tensors.H_Heis_rot(), which="horizontal")
            + tensors.E(rho, tensors.H_Heis_rot(), which="vertical")
        ) / 2

        # Check that the projector modes are correct
        assert E == pytest.approx(-0.6602310934799582, abs=1e-6)
