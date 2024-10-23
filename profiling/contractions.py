import opt_einsum as oe
import torch
import pytest

from pepsflow.models.tensors import Tensors


def CtmAlg_contractions(A):
    D = A.shape[1]
    a = torch.einsum("abcde,afghi->bfcidheg", A, A)

    # Trace out indices two obtain intial corner and edge tensors
    C = torch.einsum("aabbcdef->cdef", a).view(D**2, D**2)
    T = torch.einsum("aabcdefg->bcdefg", a).view(D**2, D**2, D**2)

    a = a.reshape(D**2, D**2, D**2, D**2)
    M = torch.einsum("ab,acd,bef,ecgh->dghf", C, T, T, a)

    # Test contractions
    path_info1 = oe.contract_path("abcde,afghi->bcdefghi", A, A)
    path_info2 = oe.contract_path("ab,acd,bef,ecgh->dghf", C, T, T, a)
    return path_info1[1].opt_cost + path_info2[1].opt_cost, M


def CtmAlg_split_contractions(A):
    D = A.shape[1]
    a = torch.einsum("abcde,afghi->bfcidheg", A, A)

    # Trace out indices two obtain intial corner and edge tensors
    C = torch.einsum("aabbcdef->cdef", a).view(D**2, D**2)
    T = torch.einsum("aabcdefg->bcdefg", a).view(D**2, D, D, D**2)

    M = torch.einsum("ab,acde,bfgh,mfcij,mglkd->eikjlh", C, T, T, A, A)

    # Test contractions
    path_info = oe.contract_path("ab,acde,bfgh,ifcjk,igdlm->ejlkmh", C, T, T, A, A)
    return path_info[1].opt_cost, M.reshape(D**2, D**2, D**2, D**2)


if __name__ == "__main__":
    D = 6
    A = Tensors.A_random_symmetric(D)
    classic_cost, classic_M = CtmAlg_contractions(A)

    split_cost, split_M = CtmAlg_split_contractions(A)

    print(f"Classic cost: {classic_cost}")
    print(f"Split cost: {split_cost}")
    print(f"Speedup: {round(classic_cost/split_cost, 3)}")
