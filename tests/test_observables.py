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
        chi, D = 16, 2
        alg = CtmSymmetric(A, chi=chi)
        alg.exe(N=100)

        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        rho = tensors.rho_symmetric(A, alg.C, alg.T)
        H = tensors.H_Ising(lam=4)
        assert tensors.E(rho, H, which="horizontal") == pytest.approx(-2.06688, abs=1e-3)

    def test_E_Heisenberg_symmetric(self):
        """
        Test the energy function with the Heisenberg model using the symmetric CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/Heis_D2_state.txt").reshape(2, 2, 2, 2, 2)).double()
        chi, D = 6, 2
        alg = CtmSymmetric(A, chi=chi)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        rho = tensors.rho_symmetric(A, alg.C, alg.T)
        H = tensors.H_Heis_rot()
        assert tensors.E(rho, H, which="horizontal") == pytest.approx(-0.6602310934799586, abs=1e-3)

    def test_E_J1J2_symmetric(self):
        """
        Test the energy function with the J1J2 model using the symmetric CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/J205_D3_state.txt").reshape(2, 3, 3, 3, 3)).double()
        chi, D = 16, 3
        alg = CtmSymmetric(A, chi=chi)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)

        rho = tensors.rho_symmetric(A, alg.C, alg.T)
        E_nn = tensors.E(rho, tensors.H_Heis_rot(), which="horizontal")
        E_nnn = tensors.E(rho, tensors.H_Heis(), which="diagonal")
        assert E_nn + 0.5 * E_nnn == pytest.approx(-0.49105775959620757, abs=1e-4)

    def test_E_J1J2_general(self):
        """
        Test the energy function with the J1J2 model using the general CTM algorithm.
        """

        A = torch.from_numpy(np.loadtxt("tests/test_states/J205_D3_state.txt").reshape(2, 3, 3, 3, 3)).double()
        chi, D = 16, 3
        alg = CtmGeneral(A, chi=chi)
        alg.exe(N=100)
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        H_rot = tensors.H_Heis_rot()
        H = tensors.H_Heis()
        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        E_nn = (tensors.E(rho, H_rot, which="horizontal") + tensors.E(rho, H_rot, which="vertical")) / 2
        E_nnn = (tensors.E(rho, H, which="diagonal") + tensors.E(rho, H, which="antidiagonal")) / 2
        assert E_nn + 0.5 * E_nnn == pytest.approx(-0.49105775959620757, abs=1e-4)
