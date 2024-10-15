from pepsflow.models.tensors import Tensors

import torch

sx = torch.Tensor([[0, 1], [1, 0]])
sz = torch.Tensor([[1, 0], [0, -1]])
sy = torch.tensor([[0, -1j], [1j, 0]])
sm = torch.Tensor([[0, 0], [1, 0]])
sp = torch.Tensor([[0, 1], [0, 0]])

H = Tensors.H_Heisenberg(lam=2, sy=sy, sz=sz, sp=sp, sm=sm)
print(H)
print("\n")
H = Tensors.H_Ising(lam=2, sz=sz, sx=sx, I=torch.eye(2))
print(H)
