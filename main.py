from tensors import Tensors
from CTM_alg import CtmAlg


def energy(C, T, a):
    pass


if __name__ == "__main__":

    a = Tensors.a_solution()
    alg = CtmAlg(a, chi=4, d=2)
    alg.exe(count=10, tol=1e-10)
