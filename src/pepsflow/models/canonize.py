from pepsflow.ipeps.ipeps import iPEPS

import torch


def canonize(ipeps: iPEPS, tolerance: float = 1e-16) -> tuple[torch.Tensor, list]:
    """Compute the minimal canonical form of the iPEPS tensor according to the algorithm 1 in
    https://arxiv.org/pdf/2209.14358v1.

    Args:
        ipeps (iPEPS): iPEPS model to canonize.
        tolerance (float): Tolerance for the convergence of the algorithm. Default is 1e-16.

    Returns:
        tuple[torch.Tensor, list]: Tuple containing the canonized iPEPS tensor and the list of
        norm (see paper for how this is defined) values at each iteration.
    """

    A = ipeps.params[ipeps.map] if ipeps.map is not None else ipeps.params
    g1 = g2 = ipeps.tensors.identity(ipeps.args["D"])
    y = []
    while True:
        A = ipeps.tensors.gauge_transform(A, g1, g2)
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
        y.append(f.item())
        if f < tolerance:
            break

        g1 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff1)
        g2 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff2)

    return A, y
