import torch


def canonize(A: torch.Tensor, tolerance: float = 1e-16) -> torch.Tensor:
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

        if (1 / trace_rho) * (diff1.norm() ** 2 + diff2.norm() ** 2) < tolerance:
            break

        g1 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff1)
        g2 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff2)

    return A
