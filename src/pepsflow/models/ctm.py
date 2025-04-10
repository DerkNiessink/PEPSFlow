from abc import ABC, abstractmethod
import torch
import scipy.sparse.linalg

from pepsflow.models.tensors import Methods
from pepsflow.models.svd import CustomSVD
from pepsflow.models.truncated_svd import truncated_svd_gesdd

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
        tensors: tuple[torch.Tensor, ...] = None,
        split: bool = False,
        iterative: bool = False,
    ):
        """
        Args:
            A (torch.Tensor): Rank-5 input tensor of shape (d, d, d, d, D).
            chi (int): Maximum bond dimension of the CTM algorithm.
            tensors (tuple): Tuple containing the corner and edge tensors of the CTM algorithm.
            split (bool): Whether to use the split or classic CTM algorithm. Default is False.
            iterative (bool): Whether to use iterative methods for the eigenvalue decomposition. Default is False.
        """
        D = A.size(1)
        self.D = D
        self.split = split
        self.max_chi = chi
        self.tensors = tensors
        self.chi = D**2 if tensors is None else tensors[0].size(0)  # In both cases we have to let chi grow to max_chi.
        self.eigvals_sums = [0]
        self.iterative = iterative

        self.a_split = torch.einsum("abcde,afghi->bfcgdhei", A, A)
        #      /
        #  -- o --
        #    /|/      🡺   [D, D, D, D, D, D, D, D]
        #  -- o --
        #    /

        self.a = A if split else self.a_split.reshape(D**2, D**2, D**2, D**2)
        #      /                                |
        #  -- o --  [d, D, D, D, D]    OR    -- o --  [D², D², D², D²]
        #    /|                                 |

    def exe(self, N: int = 1, tol: float = 1e-13):
        """
        Execute the CTM algorithm for N steps.

        `N` (int): number of steps to execute the algorithm for. Default is 1.
        `tol` (float): convergence tolerance. The algorithm is terminated when this tolerance is reached. Default is 1e-9.
        """
        self.Niter = N
        for i in range(N):
            self._step()
            if self._converged(tol):
                print("Converged after", i, "iterations.")
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C, T = self.tensors or (None, None)
        self.C = torch.einsum("aabbcdef->cdef", self.a_split).view(self.D**2, self.D**2) if C is None else C
        #       /|
        #  --- o ---
        #  |  /|/      🡺   o --  [χ, χ]
        #  --- o ---        |
        #     /

        if T is not None:
            self.T = T.view(self.chi, self.D, self.D, self.chi) if self.split else T
        else:
            shape = (self.D**2, self.D, self.D, self.D**2) if self.split else (self.D**2, self.D**2, self.D**2)
            self.T = torch.einsum("aabcdefg->bcdefg", self.a_split).view(shape)
        #       /
        #  --- o --         |                         | __
        #  |  /|/      🡺   o --  [χ, D², χ]    OR    o --  [χ, D, D, χ]
        #  --- o --         |                         |
        #   


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

        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        U = U[:, torch.argsort(torch.abs(s), descending=True)[: self.chi]]
        s = s[torch.argsort(torch.abs(s), descending=True)[: self.chi]]

        # Reshape U back in a rank-3 or 4 tensor.
        shape = (k, self.D, self.D, self.chi) if self.split else (k, self.D**2, self.chi)

        # Save the sum of the eigenvalues for convergence check.
        self.eigvals_sums.append(torch.sum(s))

        return U.view(shape)


class CtmGeneral(Ctm):
    """
    Class for the general Corner Transfer Matrix (CTM) algorithm without rotational symmetry.
    """

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

        if self.tensors:
            self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4 = self.tensors
        else:
            self.C1 = torch.einsum("abca->bc", self.a)
            self.C2 = torch.einsum("aabc->bc", self.a)
            self.C3 = torch.einsum("baac->bc", self.a) 
            self.C4 = torch.einsum("bcaa->bc", self.a) 
            #   __            __                              [χ, χ]   [χ, χ]   [χ, χ]   [χ, χ]
            #  |  |          |  |       |          |                                |      |
            #  -- o -- ,  -- o -- ,  -- o -- ,  -- o --   🡺   o -- ,   -- o ,   -- o ,    o --   
            #     |          |          |__|    |__|           |           |
     
            self.T1 = torch.einsum("abcd->bcd", self.a)
            self.T2 = torch.einsum("abcd->acd", self.a)
            self.T3 = torch.einsum("abcd->abd", self.a) 
            self.T4 = torch.einsum("abcd->abc", self.a)
            #    ___                                            [χ, D², χ]   [χ, χ, D²]   [D², χ, χ]   [χ, D², χ]
            #    \ /         | __        |        __ |                           |             |          |
            #  -- o -- ,  -- o/__| ,  -- o -- ,  |__\o --   🡺   -- o -- ,    -- o ,        -- o -- ,     o --   
            #     |          |          /_\          |              |            |                        |                         

        self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4 = [0], [0], [0], [0]


    def _step(self) -> None:

        upper_left = torch.einsum("ab,cda,bef,dghe->cghf",self.C1, self.T1, self.T4, self.a)
        upper_right = torch.einsum("ab,bdc,aef,dfgh->chge",self.C2, self.T1, self.T2, self.a)
        lower_left = torch.einsum("ab,cdb,efa,ghcf->dhge",self.C4, self.T3, self.T4, self.a)
        lower_right = torch.einsum("ab,cbd,eaf,gfch->dhge",self.C3, self.T3, self.T2, self.a)
        #  C1 -- T1 --     -- T1-- C2        
        #  |     |            |    |          __ _   ____  
        #  T4 -- a --      -- a -- T2        |__|_   _|__|
        #  |     |            |    |         |  |     |  |  
        #                               🡺                   4 x [χ, D², D², χ]  
        #  |     |            |    |         |__|_   _|__|
        #  T4 -- a --      -- a -- T2        |__|_   _|__|
        #  |     |            |    |                     
        #  C4 -- T3 --     -- T3 --C3   


        R1 = torch.einsum("abc,debc->dea", lower_left.reshape(self.chi*self.D**2, self.D**2, self.chi), upper_left)
        R1_tilde = torch.einsum("abc,debc->dea", lower_right.reshape(self.chi*self.D**2, self.D**2, self.chi), upper_right)
        #   __ _     ____          
        #  |__|_     _|__|           __ _     _ __
        #  |  |       |  |          |  |_     _|  |
        #  .  .       .  .   🡺   --|  |       |  |--
        #  |__|_     _|__|          |__|       |__|   
        #  |__|_     _|__|     [χ, D², D²χ]   [χ, D², D²χ]
   

        R2 = torch.einsum("abc,abde->edc", upper_left.reshape(self.chi, self.D**2, self.chi*self.D**2), upper_right)
        R2_tilde = torch.einsum("abc,abde->edc", lower_left.reshape(self.chi, self.D**2, self.chi*self.D**2), lower_right) 
        #   __ _ . ____          __|__
        #  |__|_ . _|__|        |_____|  [χ, D², D²χ]
        #  |  |     |  |            | |
        #        
        #                   🡺   ___|_|
        #  |__|_ . _|__|        |_____|  [χ, D², D²χ]
        #  |__|_ . _|__|           |        

        R3 = torch.einsum("abc,debc->dea", upper_left.reshape(self.chi*self.D**2, self.D**2, self.chi), lower_left)
        R3_tilde = torch.einsum("abc,debc->dea", upper_right.reshape(self.chi*self.D**2, self.D**2, self.chi), lower_right)
        #   __ _     ____          
        #  |__|_     _|__|           __         __
        #  |  |       |  |          |  |       |  |
        #  .  .       .  .   🡺   --|  |_     _|  |--
        #  |__|_     _|__|          |__|_     _|__|   
        #  |__|_     _|__|    [χ, D², D²χ]   [χ, D², D²χ]
                          

        R4 = torch.einsum("abc,abde->edc", upper_right.reshape(self.chi, self.D**2, self.chi*self.D**2), upper_left)
        R4_tilde = torch.einsum("abc,abde->edc", lower_right.reshape(self.chi, self.D**2, self.chi*self.D**2), lower_left)
        #   __ _ . ____          __|__
        #  |__|_ . _|__|        |_____|  [χ, D², D²χ]
        #  |  |     |  |        | |
        #        
        #                   🡺  |_|___
        #  |__|_ . _|__|        |_____|  [χ, D², D²χ]
        #  |__|_ . _|__|           |        

        grown_chi = min(self.chi * self.D**2, self.max_chi)

        P1, P1_tilde, sum_s1 = self._new_P(R1, R1_tilde, grown_chi)                                   
        P2, P2_tilde, sum_s2 = self._new_P(R2, R2_tilde, grown_chi)
        P3, P3_tilde, sum_s3 = self._new_P(R3, R3_tilde, grown_chi)
        P4, P4_tilde, sum_s4 = self._new_P(R4, R4_tilde, grown_chi)
        # All of shape [χ, D², χ]
    
        self.sv_sums1.append(sum_s1), self.sv_sums2.append(sum_s2), self.sv_sums3.append(sum_s3), self.sv_sums4.append(sum_s4)
        self.chi = grown_chi

        T1 = norm(torch.einsum("abc,dea,efgb,dfh->hgc", P1, self.T1, self.a, P1_tilde)) # [χ, D², χ]
        T2 = norm(torch.einsum("abc,ade,befg,dfh->chg", P2, self.T2, self.a, P2_tilde)) # [χ, χ, D²]
        T3 = norm(torch.einsum("abc,dea,fgdb,egh->fhc", P3, self.T3, self.a, P3_tilde)) # [D², χ, χ]
        T4 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P4, self.T4, self.a, P4_tilde)) # [χ, D², χ]

        C1 = norm(torch.einsum("abc,abde,edf->cf", P1_tilde, upper_left, P4_tilde)) # [χ, χ]
        C2 = norm(torch.einsum("abc,abde,edf->fc", P1, upper_right, P2_tilde))      # [χ, χ] 
        C3 = norm(torch.einsum("abc,abde,edf->fc", P3, lower_right, P2))            # [χ, χ]
        C4 = norm(torch.einsum("abc,abde,edf->fc", P3_tilde, lower_left, P4))       # [χ, χ]
        #  C1 --T1 --|\    /|-- T1--|\    /|-- T1 --C2
        #  |     |   P1~--|P1   |   P1~--P1|   |    |
        #  T4 -- a --|/    \|-- a --|/    \|-- a -- T2
        #  |____|               |              |____| 
        #  \_P4~/               .              \_P2~/                   [χ, χ]       [χ, D², χ]      [χ, χ]
        #   __|_                .               _|__                     C1 -- . . . -- T1 -- . . .  -- C2
        #  /_P4_\               .              /_P2_\                    |              |               |  
        #  |    |               |              |    |                    .              .               .
        #  T4-- a --  . . .  -- a -- . . .  -- a -- T2   🡺              .              .               .     
        #  |____|               |              |____|                    |              |               |
        #  \_P4~/               .              \_P2~/         [χ, D², χ] T4 --. . .  -- a -- . . .   -- T2 [χ, χ, D²]  
        #   _|__                .               __|_                     |              |               |
        #  /_P4_\               .              /_P2_\                    .              .               .
        #  |    |               |              |    |                    .              .               .
        #  T4 -- a --|\    /|-- a --|\    /|-- a -- T2                   |              |               |
        #  |     |   P3~--|P3   |   P3~--|P3   |    |                    C4 -- . . . -- T3 -- . . .  -- C3   
        #  C4 --T3 --|/    \|-- T3--|/    \|-- T3-- C3                  [χ, χ]      [D², χ, χ]       [χ, χ]
        self.T1, self.T2, self.T3, self.T4 = T1, T2, T3, T4
        self.C1, self.C2, self.C3, self.C4 = C1, C2, C3, C4

    def _converged(self, tol) -> bool:
        return all(abs(sv_sums[-1] - sv_sums[-2]) < tol for sv_sums in [self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4])
    
    def _new_P(self, R: torch.Tensor, R_tilde: torch.Tensor, grown_chi: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the the projector `P` and `P~`.

        Args:
            R (torch.Tensor): First half of the tensor network [χ, D², D²χ].
            R_tilde (torch.Tensor): Second half of the tensor network [χ, D², D²χ].
            grown_chi (int): The new bond dimension.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the projector `P`, `P~`, 
            and the sum of the singular values.    
        """
        A = torch.einsum("abc,abd->cd", R, R_tilde)
        #     __     __
        #    |  |---|  |
        #  --|  |   |  |--   🡺   --o--   [χD², χD²]
        #    |__|---|__|
        #U, s, Vh = CustomSVD.apply(A)
        U,s, Vh = truncated_svd_gesdd(A, grown_chi)
        Vh = Vh.T


        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]
        #U, s, Vh = U[:, :grown_chi], s[:grown_chi], Vh[:grown_chi, :]
        # --<|---o---|>--   🡺   [χD², χ], [χ], [χ, χD²]
        s_nz= s[s/s[0] > 1e-10]
        s_rsqrt= s*0
        s_rsqrt[:s_nz.size(0)]= torch.rsqrt(s_nz)
        s_rsqrt = torch.rsqrt(s)
        P_tilde = torch.einsum("abc,cd,de->abe", R_tilde, Vh.T, torch.diag(s_rsqrt))
        P = torch.einsum("ab,bc,dec->dea", torch.diag(s_rsqrt), U.T, R)
        #    ___                                                 ___            
        #  --|  |        |\                           /|        |  |--        --|\           /|--
        #    |  | ------ | | ---- o -- . . -- o ---- | | ------ |  |    🡺      | |-- . . --| |
        #  --|__|        |/                           \|        |__|--        --|/           \|--
        #              
        #    R~           V      s^(-1/2)  s^(-1/2)        U†        R          P~            P
        # [χD², D², χ]  [χD², χ]  [χ, χ]   [χ, χ]   [χ, χD²]  [χD², D², χ]    [χ, D², χ]   [χ, D², χ]

        return P, P_tilde, torch.sum(s)
