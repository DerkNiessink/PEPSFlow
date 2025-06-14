import pytest

from pepsflow.models.ctm import CtmGeneral
from pepsflow.models.tensors import Tensors
from pepsflow.ipeps.io import IO


class TestNNN:

    def test_nnn(self):
        """
        Test the energy function with the Ising model using the symmetric CTM algorithm.
        """
        chi, D = 16, 3
        tensors = Tensors(dtype="double", device="cpu", chi=chi, D=D)
        H = tensors.H_Heis_rot()

        ipeps = IO.load("tests/test_data/J102_D3_bad.json")
        A = ipeps.params
        alg = CtmGeneral(A, chi=16)
        alg.exe(N=100)
        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        Eh = tensors.E(rho, H, which="horizontal").item()
        Ev = tensors.E(rho, H, which="vertical").item()
        Ed = tensors.E(rho, H, which="diagonal").item()
        Ead = tensors.E(rho, H, which="antidiagonal").item()

        A = A.permute(0, 4, 3, 2, 1)
        alg = CtmGeneral(A, chi=chi)
        alg.exe(N=100)

        rho = tensors.rho_general(A, alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4)
        Eh2 = tensors.E(rho, H, which="horizontal").item()
        Ev2 = tensors.E(rho, H, which="vertical").item()
        Ed2 = tensors.E(rho, H, which="diagonal").item()
        Ead2 = tensors.E(rho, H, which="antidiagonal").item()

        assert Eh == pytest.approx(Ev2, abs=1e-4)
        assert Ev == pytest.approx(Eh2, abs=1e-4)
        assert Ed == pytest.approx(Ead2, abs=1e-4)
        assert Ead == pytest.approx(Ed2, abs=1e-4)
