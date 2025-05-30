import torch

from pepsflow.models.tensors import Tensors
from pepsflow.models.canonize import apply_simple_update
from pepsflow.ipeps.io import IO
from pepsflow.ipeps.tools import Tools
from pepsflow.models.ctm import CtmGeneral


class TestMinimalCanonical:

    def test_simple_update_gauge_MPS(self):
        """
        Tests if the canonical form of a MPS is achieved using the algorithm in Fig. 2 from
        https://journals.aps.org/prb/pdf/10.1103/PhysRevB.91.115137.
        """

        tensors = Tensors(dtype="double", device="cpu")
        M = tensors.random_tensor((2, 3, 3))
        g = tensors.identity(3)
        test = []
        for _ in range(100):
            rho = torch.einsum("plr,pLR->lLrR", M, M)
            #  -- o --
            #     |
            #  -- o --

            Vl_T = torch.einsum("lLrR,la,aL->Rr", rho, g, g)
            #   --o-- o --
            #   |     |
            #   --o-- o --
            S, W = torch.linalg.eigh(Vl_T)
            Yt = torch.diag(torch.sqrt(S)) @ W.T

            Vr = torch.einsum("lLrR,ra,aR->lL", rho, g, g)
            test.append(torch.linalg.norm(Vr / Vr.max() - torch.eye(3, dtype=Vr.dtype)))

            #  -- o --o--
            #     |     |
            #  -- o --o--
            S, W = torch.linalg.eigh(Vr)
            X = W @ torch.diag(torch.sqrt(S))

            U, S, V = torch.linalg.svd(Yt @ g @ X)
            S = S / S.max()

            new_g = torch.diag(S)
            new_M = torch.einsum("Ll,plr,rR->pLR", V @ torch.linalg.inv(X), M, torch.linalg.inv(Yt) @ U)
            new_M = new_M / new_M.max()
            #     V   X^-1  M  Yt^-1  U
            #  -- o -- o -- o -- o -- o --   🡺   -- o --
            #               |                        |

            M = new_M
            g = new_g
        # plt.plot(range(len(test)), test, marker="o", markersize=3)
        # plt.show()
        assert torch.allclose(Vr / Vr.max(), torch.eye(3, dtype=Vr.dtype), atol=1e-6)

    def test_minimal_canonical_MPS(self):
        """
        Tests if the minimal canonical form of a MPS is achieved using algorithm 1 from
        https://arxiv.org/pdf/2209.14358v1.
        """
        tensors = Tensors(dtype="double", device="cpu")
        M = tensors.random_tensor((2, 3, 3))
        g = tensors.identity(3)

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

        assert f < 1e-16

    def test_simple_update_PEPS(self):
        """
        Tests if the canonical form of a PEPS is achieved using the algorithm in Fig. 6 from
        https://journals.aps.org/prb/pdf/10.1103/PhysRevB.91.115137.
        """

        tensors = Tensors(dtype="double", device="cpu")
        ipeps = IO.load("tests/test_data/D3_test.json")
        A = ipeps.params
        alg = CtmGeneral(A, chi=32, projector_mode="svd")
        alg.exe(N=20)
        E_h = tensors.E_horizontal_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_v = tensors.E_vertical_nn_general(
            A, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_without_gauge = (E_h + E_v) / 2

        A, gh, gv = apply_simple_update(A, tolerance=1e-20, separated=True)

        # Test the conditions in Fig. 5 of the paper
        rho = torch.einsum("purdl,pURDL->urdlURDL", A, A)
        Vl_T = torch.einsum("urdlURDL,ua,aU,db,bD,lc,cL->Rr", rho, gv, gv, gv, gv, gh, gh)
        Vr = torch.einsum("urdlURDL,ua,aU,rb,bR,dc,cD->lL", rho, gv, gv, gh, gh, gv, gv)
        Vu = torch.einsum("urdlURDL,ra,aR,db,bD,lc,cL->Uu", rho, gh, gh, gv, gv, gh, gh)
        Vd_T = torch.einsum("urdlURDL,ua,aU,rb,bR,lc,cL->dD", rho, gv, gv, gh, gh, gh, gh)
        id = torch.eye(Vl_T.shape[0], dtype=Vl_T.dtype)
        for mat, name in zip([Vl_T, Vr, Vu, Vd_T], ["Vl_T", "Vr", "Vu", "Vd_T"]):
            norm_mat = mat / mat.max()
            assert torch.allclose(norm_mat, id, atol=1e-10), f"{name} is not close to identity"

        # check that the energy is the same with and without gauge
        A_gauged = torch.einsum("purdl,dD,lL->purDL", A, gv, gh)
        alg = CtmGeneral(A_gauged, chi=32, projector_mode="svd")
        alg.exe(N=20)
        E_h = tensors.E_horizontal_nn_general(
            A_gauged, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_v = tensors.E_vertical_nn_general(
            A_gauged, tensors.H_Heis_rot(), alg.C1, alg.C2, alg.C3, alg.C4, alg.T1, alg.T2, alg.T3, alg.T4
        )
        E_with_gauge = (E_h + E_v) / 2
        print(f"Energy without gauge: {E_without_gauge}, Energy with gauge: {E_with_gauge}")
        assert abs(E_with_gauge - E_without_gauge) < 1e-7, "Energy with and without gauge differ too much"

    def test_minimal_canonical_PEPS(self):
        """
        Tests if the minimal canonical form of a PEPS is achieved by using Tools.minimize_norm and is
        saved and loaded correctly.
        """

        ipeps = IO.load("tests/test_data/test.json")
        Tools.gauge(ipeps, args={"tolerance": 1e-16, "gauge": "minimal_canonical", "seed": 20})
        IO.save(ipeps, "tests/test_data/test_minimal_canonical.json")
        ipeps = IO.load("tests/test_data/test_minimal_canonical.json")

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
