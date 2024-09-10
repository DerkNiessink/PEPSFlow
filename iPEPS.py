import torch

from models.tensors import Methods
from models.CTM_alg import CtmAlg
from models.energy import get_energy


class iPEPS(torch.nn.Module):
    def __init__(self, chi, d, H):
        super(iPEPS, self).__init__()
        self.chi = chi
        self.d = d
        self.H = H

        # A(phy, up, left, down, right)
        A = torch.randn(d, d, d, d, d).double()
        A = A / A.norm()
        self.A = torch.nn.Parameter(A)

    def forward(self):

        Asymm = Methods.symmetrize_rank5(self.A)
        d, chi, H = self.d, self.chi, self.H
        T = (
            (Asymm.view(d, -1).t() @ Asymm.view(d, -1))
            .contiguous()
            .view(d, d, d, d, d, d, d, d)
        )
        T = T.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(d**2, d**2, d**2, d**2)
        T = T / T.norm()

        alg = CtmAlg(T, chi, d)
        alg.exe()
        C, E = alg.C, alg.T
        loss = get_energy(Asymm, H, C, E)

        return loss
