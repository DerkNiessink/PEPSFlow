import pytest
import torch
import matplotlib.pyplot as plt

from pepsflow.models.tensors import Tensors
from pepsflow.ipeps.io import IO
from pepsflow.ipeps.tools import Tools


class TestMinimalCanonical:

    def test_minimal_canonical_MPS(self):
        """
        Tests if the minimal canonical form of a MPS is achieved using algorithm 1 from
        https://arxiv.org/pdf/2209.14358v1.
        """

        tensors = Tensors(dtype="double", device="cpu")
        M = tensors.random_tensor((2, 3, 3))
        g = tensors.identity(3)
        y = []
        for _ in range(1000):
            g_inv = torch.linalg.inv(g)
            M = torch.einsum("al,prl,rb->pab", g, M, g_inv)
            #   --<|-- o --|>--
            #          |

            rho = torch.einsum("plr,pLR->lLrR", M, M)
            #  -- o --
            #     |
            #  -- o --

            rho_1 = torch.einsum("llrR->rR", rho)
            #  -- o --
            #  |  |
            #  -- o --

            rho_2 = torch.einsum("lLrr->lL", rho)
            #  -- o --
            #     |   |
            #  -- o --

            diff = rho_1 - rho_2.T
            trace_rho = torch.einsum("llrr->", rho)
            #  -- o --
            #  |  |  |
            #  -- o --
            f = (1 / trace_rho) * diff.norm() ** 2
            y.append(f.item())

            g = torch.linalg.matrix_exp((-1 / (4 * trace_rho)) * diff)

        gA = torch.einsum("prl,Plr->pP", M, M)
        #     g       g^-1  g        g-1
        #  --<|-- o --|>--<|-- o --|>--
        #         |            |
        #
        A = torch.einsum("prl,Plr->pP", M, M)

        # plt.plot(range(len(y)), y, marker="o", markersize=3)
        # plt.xlabel("Iteration")
        # plt.yscale("log")
        # plt.ylabel("Norm")
        # plt.show()

        assert gA == pytest.approx(A, rel=1e-5) and (f < 1e-16)

    def test_minimal_canonical_PEPS(self):
        """
        Tests if the minimal canonical form of a PEPS is achieved by using Tools.minimize_norm and is
        saved and loaded correctly.
        """

        ipeps = IO.load("tests/test.json")
        Tools.gauge(ipeps, args={"tolerance": 1e-16, "gauge": "minimal_canonical"})
        IO.save(ipeps, "tests/test_minimal_canonical.json")
        ipeps = IO.load("tests/test_minimal_canonical.json")

        A = ipeps.params
        rho = torch.einsum("purdl,pURDL->urdlURDL", A, A)
        rho_11 = torch.einsum("urdluRdl->rR", rho)
        rho_12 = torch.einsum("urdlurdL->lL", rho)
        rho_21 = torch.einsum("urdlUrdl->uU", rho)
        rho_22 = torch.einsum("urdlurDl->dD", rho)
        trace_rho = torch.einsum("urdlurdl->", rho)
        diff1 = rho_11 - rho_12.T
        diff2 = rho_21 - rho_22.T

        f = (1 / trace_rho) * (diff1.norm() ** 2 + diff2.norm() ** 2)
        assert f < 1e-16
