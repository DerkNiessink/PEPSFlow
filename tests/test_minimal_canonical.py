import pytest
import torch
import matplotlib.pyplot as plt

from pepsflow.models.tensors import Tensors


def test_minimal_canonical_MPS():
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
        # if f < args["tolerance"]:
        #    break

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
