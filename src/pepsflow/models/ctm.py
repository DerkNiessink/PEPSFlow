from abc import ABC, abstractmethod
import torch
import scipy.sparse.linalg
import matplotlib.pyplot as plt

from pepsflow.models.tensors import Methods

norm = Methods.normalize
symm = Methods.symmetrize

# fmt: off

class Ctm(ABC):
    """
    Abstract base class for the Corner Transfer Matrix (CTM) algorithm.
    """

    #            /
    #   A =  -- o --  [D, d, d, d, d]
    #          /|
    #
    #   C =  o --  [χ, χ]
    #        |
    #
    #        |
    #   T =  o --  [χ, D², χ]
    #        |

    def __init__(
        self,
        A: torch.Tensor,
        chi: int,
        C: torch.Tensor = None,
        T: torch.Tensor = None,
        split: bool = False,
        iterative: bool = False,
    ):
        """
        Args:
            A (torch.Tensor): Rank-5 input tensor of shape (d, d, d, d, D).
            chi (int): Maximum bond dimension of the CTM algorithm.
            C (torch.Tensor): Corner tensor of the iPEPS tensor network.
            T (torch.Tensor): Edge tensor of the iPEPS tensor network.
            split (bool): Whether to use the split or classic CTM algorithm. Default is False.
            iterative (bool): Whether to use iterative methods for the eigenvalue decomposition. Default is False.
        """
        D = A.size(1)
        self.D = D
        self.split = split
        self.max_chi = chi
        self.chi = D**2 if C is None else C.size(0)  # In both cases we have to let chi grow to max_chi.
        self.eigvals_sums = [0]
        self.iterative = iterative

        a = torch.einsum("abcde,afghi->bfcidheg", A, A)
        #      /
        #  -- o --
        #    /|/      🡺   [D, D, D, D, D, D, D, D, D]
        #  -- o --
        #    /

        self.C = torch.einsum("aabbcdef->cdef", a).view(D**2, D**2) if C is None else C
        #       /|
        #  --- o ---
        #  |  /|/      🡺   o --  [χ, χ]
        #  --- o ---        |
        #     /

        if T is not None:
            self.T = T.view(self.chi, D, D, self.chi) if split else T
        else:
            shape = (D**2, D, D, D**2) if split else (D**2, D**2, D**2)
            self.T = torch.einsum("aabcdefg->bcdefg", a).view(shape)
        #       /
        #  --- o --         |                         | __
        #  |  /|/      🡺   o --  [χ, D², χ]    OR    o --  [χ, D, D, χ]
        #  --- o --         |                         |
        #     /

        self.a = A if split else a.reshape(D**2, D**2, D**2, D**2)
        #      /                                |
        #  -- o --  [d, D, D, D, D]    OR    -- o --  [D², D², D², D²]
        #    /|                                 |

    def exe(self, N: int = 1, tol: float = 1e-9):
        """
        Execute the CTM algorithm for N steps.

        `N` (int): number of steps to execute the algorithm for. Default is 1.
        `tol` (float): convergence tolerance. The algorithm is terminated when this tolerance is reached. Default is 1e-9.
        """
        self.Niter = N
        for i in range(N):
            self._step()
            if self._converged(tol):
                self.Niter = i
                break

        if self.split:
            self.T = self.T.view(self.chi, self.D**2, self.chi)

    @abstractmethod
    def _step(self) -> None:
        pass

    @abstractmethod
    def _converged(self, tol: float) -> bool:
        pass


class CtmSymmetric(Ctm):
    """
    Class for the rotational symmetric Corner Transfer Matrix (CTM) algorithm.
    """

    def _step(self) -> None:
        self._split_step() if self.split else self._classic_step()

    def _converged(self, tol: float) -> bool:
        return abs(self.eigvals_sums[-1] - self.eigvals_sums[-2]) < tol

    def _classic_step(self):
        """
        Execute one "classic" CTM step. This is the standard CTM algorithm for the rank-4 input tensor.
        """
        M = torch.einsum("ab,acd,bef,ecgh->dgfh", self.C, self.T, self.T, self.a)
        #   o -- o --
        #   |    |      🡺   [χ, D², χ, D²]
        #   o -- o --
        #   |    |

        U = self._new_U(M)
        #  --|\
        #    | |--   🡺   [χ, D², χ]
        #  --|/
        #

        self.C = symm(norm(torch.einsum("abc,abfe,fed->cd", U, M, U)))
        #  o -- o --|\
        #  |    |   | |--
        #  o -- o --|/      🡺   o --   [χ, χ]
        #  |____|                |
        #  \____/
        #     |

        self.T = symm(norm(torch.einsum("cba,cgf,bdeg,feh->adh", U, self.T, self.a, U)))
        #   _|__
        #  /____\
        #  |    |           |
        #  o -- o --   🡺   o --  [χ, D², χ]
        #  |____|           |
        #  \____/
        #     |

    def _split_step(self):
        """
        Execute one "split" CTM step. This is the CTM algorithm for the rank-5 input tensor.
        """
        # fmt: off
        M = torch.einsum("ab,acde,bfgh,mfcij,mglkd->eikhjl", self.C, self.T, self.T, self.a, self.a)
        #        o----o----
        #       /    /|
        #      /_- o---- 
        #     //  /|/      🡺   [χ, D, D, χ, D, D]     
        #    o---/-o---- 
        #   /     /
        #  /     /

        U = self._new_U(M)   
        #  --|\   
        #  --| |--   🡺   [χ, D, D, χ]
        #  --|/    
                                                                       
        self.C = symm(norm(torch.einsum("abcd,abcefh,efhg->dg", U, M, U)))
        #        o----o-_
        #       /    /| |\
        #      /_- o---_| |--        
        #     //  /|/   |/      🡺   o --  [χ, χ]
        #    o---/-o---/             |
        #    /___|_/         
        #    \____/
        #      /

        self.T = symm(norm(torch.einsum("abcd,aefg,lbemh,lcfij,gmik->dhjk", U, self.T, self.a, self.a, U)))
        #        __/_
        #       /____\
        #      /   / /        | __
        #     /_- o---   🡺   o --  [χ, D, D, χ]
        #    //  /|/          |
        #   o---/ o --
        #  /___|_/
        #  \____/
        #    /

    def _new_U(self, M: torch.Tensor) -> torch.Tensor:
        """
        Return a tuple of the truncated `U` tensor and `s` matrix, by conducting a singular value decomposition
        (svd) on the given corner tensor `M`. Using this factorization `M` can be written as M = U s V*, where
        the `U` matrix is used for renormalization.

        `M` (torch.Tensor): The new contracted corner tensor of shape (chi, d, chi, d).

        Returns the renormalization tensor of shape (chi, d, chi) which is obtained by reshaping `U` in a rank-3.
        """
        M = M.contiguous().view(self.chi * self.D**2, self.chi * self.D**2)
        #  --o--  [χD², χD²]

        # Let chi grow if the desired chi is not yet reached.
        k = self.chi
        self.chi = min(self.chi * self.D**2, self.max_chi)

        if self.iterative and M.device.type == "cpu":
            s, U = scipy.sparse.linalg.eigsh(M.cpu().detach().numpy(), k=self.chi)
            s, U = torch.from_numpy(s), torch.from_numpy(U)
        else:
            s, U = torch.linalg.eigh(M)
            # Sort the eigenvectors by the absolute value of the eigenvalues and keep the χ largest ones.
            U = U[:, torch.argsort(torch.abs(s), descending=True)[: self.chi]]
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        # Reshape U back in a rank-3 or 4 tensor.
        shape = (k, self.D, self.D, self.chi) if self.split else (k, self.D**2, self.chi)

        # Save the sum of the eigenvalues for convergence check.
        self.eigvals_sums.append(torch.sum(s))

        return U.view(shape)


class CtmGeneral(Ctm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, split=False, **kwargs)

        # Let's define the corner and edge tensors like this:
        #
        #  C1 -- T1 -- C2
        #  |     |     |
        #  T4 -- a  -- T2
        #  |     |     |
        #  C4 -- T3 -- C3
        #
        # And let's label the legs of the a tensor like this:
        #
        #     |
        #  -- o --  [D², D², D², D²]   🡺    [up, right, down, left]
        #     |
        #

        self.C1 = torch.einsum("abca->bc", self.a)
        self.C2 = torch.einsum("aabc->bc", self.a)
        self.C3 = torch.einsum("baac->bc", self.a)
        self.C4 = torch.einsum("bcaa->bc", self.a)
        #   __            __                              [χ, χ]   [χ, χ]   [χ, χ]   [χ, χ]
        #  |  |          |  |       |          |                                |      |
        #  -- o -- ,  -- o -- ,  -- o -- ,  -- o --   🡺   o -- ,   -- o ,   -- o ,    o --   
        #     |          |          |__|    |__|           |           |
        self.T1 = torch.einsum("abcd->bcd", self.a)
        self.T2 = torch.einsum("bacd->bcd", self.a)
        self.T3 = torch.einsum("bcad->bcd", self.a)
        self.T4 = torch.einsum("bcda->bcd", self.a)
        #    ___                                            [χ, D², χ]   [χ, χ, D²]   [D², χ, χ]   [χ, D², χ]
        #    \ /         | __        |        __ |                           |             |          |
        #  -- o -- ,  -- o/__| ,  -- o -- ,  |__\o --   🡺   -- o -- ,    -- o ,        -- o -- ,     o --   
        #     |          |          /_\          |              |            |                        |
        #                                                      

        self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4 = [0], [0], [0], [0]


    def _step(self) -> None:

        P4, P4_tilde = self._new_P2_P4() # [χ, D², χD²], [χ, D², χD²]

        # Let chi grow if the desired chi is not yet reached.
        self.chi = min(self.chi * self.D**2, self.max_chi)

        P4 = P4[:, :, : self.chi]
        P4_tilde = P4_tilde[:, :, : self.chi]

        self.T4 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P4, self.T4, self.a, P4_tilde))
        #  C1 --T1 --|\    /|-- T1--|\    /|-- T1 --C2
        #  |     |   P1|--P1|   |   P1|--P1|   |    |
        #  T4 -- a --|/    \|-- a --|/    \|-- a -- T2
        #  |____|               |              |____| 
        #  \_P4_/               .              \_P2_/                   [χ, χ]       [χ, D², χ]      [χ, χ]
        #   __|_                .               _|__                     C1 -- . . . -- T1 -- . . .  -- C2
        #  /_P4_\               .              /_P2_\                    |              |               |  
        #  |    |               |              |    |                    .              .               .
        #  T4-- a --  . . .  -- a -- . . .  -- a -- T2   🡺              .              .               .     
        #  |____|               |              |____|                    |              |               |
        #  \_P4_/               .              \_P2_/         [χ, D², χ] T4 --. . .  -- a -- . . .   -- T2 [χ, χ, D²]  
        #   _|__                .               __|_                     |              |               |
        #  /_P4_\               .              /_P2_\                    .              .               .
        #  |    |               |              |    |                    .              .               .
        #  T4 -- a --|\    /|-- a --|\    /|-- a -- T2                   |              |               |
        #  |     |   |P3--P3|   |   |P3--P3|   |    |                    C4 -- . . . -- T3 -- . . .  -- C3   
        #  C4 --T3 --|/    \|-- T3--|/    \|-- T3-- C3                  [χ, χ]      [D², χ, χ]       [χ, χ]
    
    def _converged(self, tol: float) -> bool:
        return abs(self.sv_sums4[-1] - self.sv_sums4[-2]) < tol
        #return all(abs(sv_sums[-1] - sv_sums[-2]) < tol for sv_sums in [self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4])

    def _new_P2_P4(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the new projector P4, which is used for a left move in the renormalization step.
        """

        R = torch.einsum(
            "ab,cda,bef,dghe,ijkg,lmj,nic,ln->fhkm", 
            self.C1, self.T1, self.T4, self.a, self.a, self.T2, self.T1, self.C2
        )
        #  C1 -- T1 -- T1-- C2         _____    
        #  |     |     |    |    🡺   |_____|   [χ, D², D², χ]
        #  T4 -- a --- a -- T2        | | | |
        #  |     |     |    |

        R = R.view(self.chi, self.D**2, self.chi*self.D**2)
        #   _____                           __|__
        #  |_____|   [χ, D², χ, D²]   🡺   |_____|   [χ, D², χD²]
        #  | | | |                         | |   

        R_tilde = torch.einsum(
            "ab,cda,efb,ghcf,ijkh,lmj,knd,mn->egil", 
            self.C4, self.T3, self.T4, self.a, self.a, self.T2, self.T3, self.C3
        )
        #  |     |     |    |
        #  T4 -- a --  a -- T2        |_|_|_|    
        #  |     |     |    |    🡺   |_____|   [χ, D², D², χ] 
        #  C4 -- T3 -- T3 --C3       

        R_tilde = R_tilde.view(self.chi, self.D**2, self.chi*self.D**2)
        #  |_|_|_|                         |_|___
        #  |_____|   [χ, D², χ, D²]   🡺   |_____|   [χ, D², χD²]
        #                                     |

        A = torch.einsum("abc,abd->cd", R, R_tilde)
        #   __|__  
        #  |_____|
        #  |_|___    🡺   --o--   [χD², χD²]
        #  |_____|   
        #     |  

        U, s, Vh = torch.linalg.svd(A)
        s = torch.diag(s)
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        U, s, Vh = U[:, :self.chi], s[:self.chi, :self.chi], Vh[:self.chi, :]
        # --<|---o---|>--   🡺   [χD², χ], [χ, χ], [χ, χD²]

        P4_tilde = torch.einsum("abc,cd,de->abe", R_tilde, Vh.T, torch.sqrt(torch.linalg.inv(s)))
        P4 = torch.einsum("ab,bc,dec->dea", torch.sqrt(torch.linalg.inv(s)), U.T, R)
        #  |_|___  
        #  |_____| R~       [χ, D², χD²]
        #    _|_
        #   \___/  V        [χD², χ]
        #     |                               |___|        
        #     o    s^(-1/2) [χ, χ]            \___/  P~  [χ, D², χ] 
        #     |                                 | 
        #     .                           🡺    .                     
        #     .                                 .
        #     |                                _|_
        #     o    s^(-1/2) [χ, χ]            /___\  P   [χ, D², χ] 
        #    _|_                              |   |
        #   /___\  U†       [χ, χD²]          
        #   __|__
        #  |_____| R        [χ, D², χD²]
        #  | |

        self.sv_sums4.append(torch.sum(s))

        B = torch.einsum("abc,abd,efd,efg->cg", R, P4_tilde, P4, R_tilde)
        #   __|__  
        #  |_____|
        #  |_|
        #  \_/
        #   |        🡺   --o--   [χD², χD²]
        #  /_\
        #  |_|___    
        #  |_____|   
        #     |  

        self.diff = torch.norm(A - B) / torch.norm(A)
        return P4, P4_tilde



    def _new_P2(self) -> torch.Tensor:
        pass

    def _new_P3(self) -> torch.Tensor:
        pass

    def _new_P1(self) -> torch.Tensor:
        pass
