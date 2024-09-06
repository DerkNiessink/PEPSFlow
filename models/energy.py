import torch
import numpy as np


def get_energy(Asymm: np.ndarray, H: np.ndarray, C: np.ndarray, E: np.ndarray) -> float:
    """
    Get the energy of a PEPS state.

    Args:
        Asymm (np.ndarray): Symmetric A tensor of the PEPS state.
        H (np.ndarray): Hamiltonian operator (l -> d^2, r -> d^2).
        C (np.ndarray): Corner tensor obtained in CTMRG algorithm (d -> chi, r -> chi).
        E (np.ndarray): Edge tensor obtained in CTMRG algorithm (u -> chi, d -> chi, r -> d).

    Returns:
        float: Energy of the PEPS state
    """

    # Convert to torch tensors and to the right leg order:
    # A(phy,u,l,d,r), C(d,r), E(u,r,d)
    Asymm = torch.from_numpy(Asymm)
    E = torch.from_numpy(E.transpose(0, 2, 1))
    C = torch.from_numpy(C)
    H = torch.from_numpy(H)

    Da = Asymm.size()
    Td = (
        torch.einsum("mefgh,nabcd->eafbgchdmn", (Asymm, Asymm))
        .contiguous()
        .view(Da[1] ** 2, Da[2] ** 2, Da[3] ** 2, Da[4] ** 2, Da[0], Da[0])
    )

    CE = torch.tensordot(C, E, ([1], [0]))  # C(1d)E(dga)->CE(1ga)
    EL = torch.tensordot(
        E, CE, ([2], [0])
    )  # E(2e1)CE(1ga)->EL(2ega)  use E(2e1) == E(1e2)
    EL = torch.tensordot(EL, Td, ([1, 2], [1, 0]))  # EL(2ega)T(gehbmn)->EL(2ahbmn)
    EL = torch.tensordot(
        EL, CE, ([0, 2], [0, 1])
    )  # EL(2ahbmn)CE(2hc)->EL(abmnc), use CE(2hc) == CE(1ga)

    Rho = (
        torch.tensordot(EL, EL, ([0, 1, 4], [0, 1, 4]))
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(Da[0] ** 2, Da[0] ** 2)
    )

    Rho = 0.5 * (Rho + Rho.t())

    Tnorm = Rho.trace()
    Energy = torch.mm(Rho, H).trace() / Tnorm

    return float(Energy)
