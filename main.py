from models.tensors import Tensors
from models.CTM_alg import CtmAlg
from models.energy import get_energy

import numpy as np
import matplotlib.pyplot as plt

a = Tensors.a_solution()
alg = CtmAlg(a, chi=20, d=2)
alg.exe(count=10, tol=1e-10)

energy = get_energy(Tensors.A_solution(), Tensors.H(lam=4), alg.C, alg.T)
print(energy)
