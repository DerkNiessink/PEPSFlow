import torch
import matplotlib.pyplot as plt

from pepsflow.models.tensors import Tensors
from pepsflow.models.ctm import CtmGeneral


def apply_minimal_canonical(A: torch.Tensor, tolerance: float = 1e-16) -> torch.Tensor:
    """Compute the minimal canonical form of the iPEPS tensor according to the algorithm 1 in
    https://arxiv.org/pdf/2209.14358v1.

    Args:
        A (torch.Tensor): Input iPEPS state tensor of shape [d, D, D, D, D]
        tolerance (float): Tolerance for convergence. Default is 1e-16.

    Returns:
        torch.Tensor: Minimal Canonical version of the input state of shape [d, D, D, D, D]
    """

    g1 = g2 = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    while True:
        A = torch.einsum("purdl,Uu,Rr,dD,lL->pURDL", A, g2, g1, torch.linalg.inv(g2), torch.linalg.inv(g1))
        #            g2^-1
        #            /                   /
        #    g1  -- o -- g1^-1   🡺  -- o --
        #          /|                  /|
        #       g2

        rho = torch.einsum("purdl,pURDL->urdlURDL", A, A)
        #      /
        #  -- o --
        #    /|/      🡺   [D, D, D, D, D, D, D, D]
        #  -- o --
        #    /

        rho_11 = torch.einsum("urdluRdl->rR", rho)
        rho_12 = torch.einsum("urdlurdL->lL", rho)
        # tracing over all legs except the right and left legs respectively:
        #      /|         /|
        #  -- o---    -- o---
        #  | /|/    ,   /|/ |   🡺   [D, D], [D, D]
        #  -|-o --    -|-o --
        #   |/         |/

        rho_21 = torch.einsum("urdlUrdl->uU", rho)
        rho_22 = torch.einsum("urdlurDl->dD", rho)
        # tracing over all legs except the up and down legs respectively:
        #      /           /|
        #  -- o --     -- o---
        #  | /|/ |  ,  | /|/ |   🡺   [D, D], [D, D]
        #  -|-o --     -- o --
        #   |/           /

        trace_rho = torch.einsum("urdlurdl->", rho)
        # tracing over all legs:
        #      /|
        #  -- o---
        #  | /|/ |
        #  -|-o --
        #   |/

        diff1 = rho_11 - rho_12.T
        diff2 = rho_21 - rho_22.T

        f = (1 / trace_rho) * (diff1.norm() ** 2 + diff2.norm() ** 2)
        print(f"norm: {f.item()}")
        if f < tolerance:
            break

        g1 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff1)
        g2 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff2)

    return A


def apply_simple_update(
    A: torch.Tensor, tolerance: float = 1e-16, separated: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a simple update gauge to the iPEPS tensor to achieve a canonical form. The algorithm
    is based on the one presented in Fig. 6 of https://journals.aps.org/prb/pdf/10.1103/PhysRevB.91.115137.

    Args:
        A (torch.Tensor): Input iPEPS state tensor of shape [d, D, D, D, D]
        tolerance (float): Tolerance for convergence. Default is 1e-16.
        separated (bool): If separated is True, returns the new iPEPS tensor and the horizontal and vertical
            gauge transformations separately. If False, returns the new iPEPS tensor only. Default is False.

    Returns:
        torch.Tensor: Updated iPEPS tensor of shape [d, D, D, D, D] or a tuple of (iPEPS tensor, horizontal gauge,
                      vertical gauge) if separated is True.
    """

    D = A.shape[1]
    gh = gv = torch.eye(D, dtype=A.dtype, device=A.device)
    while True:
        rho = torch.einsum("purdl,pURDL->urdlURDL", A, A)
        #      /
        #  -- o --
        #    /|/      🡺   [D, D, D, D, D, D, D, D]
        #  -- o --
        #    /

        Vl_T = torch.einsum("urdlURDL,ua,aU,db,bD,lc,cL->Rr", rho, gv, gv, gv, gv, gh, gh)
        Vr = torch.einsum("urdlURDL,ua,aU,rb,bR,dc,cD->lL", rho, gv, gv, gh, gh, gv, gv)
        # tracing over all legs except the right and left legs respectively:
        #      /|         /|
        #  -- o---    -- o---
        #  | /|/    ,   /|/ |   🡺   [D, D], [D, D]
        #  -|-o --    -|-o --
        #   |/         |/
        Vu = torch.einsum("urdlURDL,ra,aR,db,bD,lc,cL->Uu", rho, gh, gh, gv, gv, gh, gh)
        Vd_T = torch.einsum("urdlURDL,ua,aU,rb,bR,lc,cL->dD", rho, gv, gv, gh, gh, gh, gh)
        # tracing over all legs except the up and down legs respectively:
        #      /           /|
        #  -- o --     -- o---
        #  | /|/ |  ,  | /|/ |   🡺   [D, D], [D, D]
        #  -|-o --     -- o --
        #   |/           /
        Vl_T = Vl_T / Vl_T.max()
        Vr = Vr / Vr.max()
        Vu = Vu / Vu.max()
        Vd_T = Vd_T / Vd_T.max()

        def gauge_update(VL, VR, g_old):
            S1, W1 = torch.linalg.eigh(VL)
            Y_t = torch.diag(torch.sqrt(S1)) @ W1.T
            S2, W2 = torch.linalg.eigh(VR)
            X = W2 @ torch.diag(torch.sqrt(S2))
            U, S, V = torch.linalg.svd(Y_t @ g_old @ X)
            g_new = torch.diag(S)
            return g_new, torch.linalg.inv(Y_t) @ U, V @ torch.linalg.inv(X)

        gh, Ah_l, Ah_r = gauge_update(Vl_T, Vr, gh)
        gv, Av_d, Av_u = gauge_update(Vd_T, Vu, gv)

        A = torch.einsum("purdl,dD,Uu,rR,Ll->pURDL", A, Av_d, Av_u, Ah_l, Ah_r)
        trace_rho = torch.einsum("urdlurdl->", rho)

        norm = lambda M: (M / M.max() - torch.eye(D, dtype=M.dtype, device=M.device)).norm() ** 2
        f = norm(Vl_T) + norm(Vr) + norm(Vu) + norm(Vd_T)

        print(f"simultaneous norm = {f.item()}")
        if f < tolerance:
            print(f"condtions: {Vl_T, Vr, Vu, Vd_T}")
            break

    if separated:
        return A, gh, gv
    else:
        return torch.einsum("purdl,uU,rR->pURdl", A, gv, gh)
