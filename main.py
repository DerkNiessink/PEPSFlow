from models.tensors import Tensors
from models.CTM_alg import CtmAlg
from models.energy import get_energy
from models.map import map_to_A_tensor
from iPEPS import iPEPS

import torch
import numpy as np


def energy():
    A = Tensors.A_solution()
    a = Tensors.a(A)
    H = Tensors.H(lam=4)
    alg = CtmAlg(a, chi=10, d=2)
    alg.exe(count=10, tol=1e-10)
    return get_energy(A, H, alg.C, alg.T)


if __name__ == "__main__":
    model = iPEPS(chi=6, d=2, H=Tensors.H(lam=3))
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50)
    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print("total number of trainable parameters:", nparams)

    def train():
        """
        Train the parameters of the iPEPS model.
        """
        optimizer.zero_grad()
        loss = model.forward()
        loss.backward()
        return loss

    for epoch in range(100):
        loss = optimizer.step(train)
        print("epoch:", epoch, "loss:", loss.item())
