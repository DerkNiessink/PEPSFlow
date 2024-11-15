import torch

from pepsflow.models.CTM_alg import CtmAlg


def converge(A: torch.Tensor, chi: int, N: int):
    alg = CtmAlg(A, chi)
    alg.exe(N=N)
