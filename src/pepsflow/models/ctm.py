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
        #    ___                                            [χ, D², χ]   [χ, D², χ]   [χ, D², χ]   [χ, D², χ]
        #    \ /         | __        |        __ |                           |             |          |
        #  -- o -- ,  -- o/__| ,  -- o -- ,  |__\o --   🡺   -- o -- ,    -- o ,        -- o -- ,     o --   
        #     |          |          /_\          |              |            |                        |
        #                                                      

        self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4 = [0], [0], [0], [0]


    def _step(self) -> None:
        
        R_up = torch.einsum(
            "ab,adc,bef,dghe,ijkg,ljm,cin,ln->fhkm", 
            self.C1, self.T1, self.T4, self.a, self.a, self.T2, self.T1, self.C2
        )
        #  C1 -- T1 -- T1-- C2         _____    
        #  |     |     |    |    🡺   |_____|   [χ, D², D², χ]
        #  T4 -- a --- a -- T2        | | | |
        #  |     |     |    |

        R_down = torch.einsum(
            "ab,dca,efb,ghcf,ijkh,ljm,nkd,mn->egil", 
            self.C4, self.T3, self.T4, self.a, self.a, self.T2, self.T3, self.C3
        )
        #  |     |     |    |
        #  T4 -- a --  a -- T2        |_|_|_|    
        #  |     |     |    |    🡺   |_____|   [χ, D², D², χ] 
        #  C4 -- T3 -- T3 --C3       

        R_left = torch.einsum(
            "ab,bcd,aef,eghc,hijk,ljm,dkn,nm->fgil",
            self.C1, self.T4, self.T1, self.a, self.a, self.T3, self.T4, self.C4
        )
        #  C1 -- T1--  
        #  |     |            ____
        #  T4 -- a --        |  |_
        #  |     |      🡺   |  |_   [χ, D², D², χ] 
        #  T4 -- a --        |__|_
        #  |     |            
        #  C4 -- T3--

        R_right = torch.einsum(
            "ab,bdc,efa,fdgh,gijk,ljm,cin,nl->ehkm",
            self.C2, self.T2, self.T1, self.a, self.a, self.T3, self.T2, self.C3
        )
        #  -- T1 --C2
        #     |    |         ____
        #  -- a -- T2        _|  |
        #     |    |    🡺   _|  |   [χ, D², D², χ] 
        #  -- a -- T2        _|__|
        #     |    |
        #  -- T3 --C3

        grown_chi = min(self.chi * self.D**2, self.max_chi)
       
        P1, P1_tilde = self._new_P1(R_right, R_left, grown_chi)
        P2, P2_tilde = self._new_P2(R_up, R_down, grown_chi)    
        P3, P3_tilde = self._new_P3(R_right, R_left, grown_chi)
        P4, P4_tilde = self._new_P4(R_up, R_down, grown_chi)    
        # All projectors are of shape [χ, D², χ]

        self.chi = grown_chi

        T1 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P1, self.T1, self.a, P1_tilde))
        T2 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P2, self.T2, self.a, P2_tilde))
        T3 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P3, self.T3, self.a, P3_tilde))
        T4 = norm(torch.einsum("abc,ade,bfgd,egh->cfh", P4, self.T4, self.a, P4_tilde))
        C1 = norm(torch.einsum("abc,ade,ef,hgf,bigd,hij->cj", P1, self.T1, self.C1, self.T4, self.a, P4))
        C2 = norm(torch.einsum("abc,ade,ef,hgf,bigd,hij->cj", P2, self.T2, self.C2, self.T1, self.a, P1))
        C3 = norm(torch.einsum("abc,ade,ef,hgf,bigd,hij->cj", P3, self.T3, self.C3, self.T2, self.a, P2))
        C4 = norm(torch.einsum("abc,ade,ef,hgf,bigd,hij->cj", P4, self.T4, self.C4, self.T3, self.a, P3))
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
        #  \_P4_/               .              \_P2_/         [χ, D², χ] T4 --. . .  -- a -- . . .   -- T2 [χ, D², χ]  
        #   _|__                .               __|_                     |              |               |
        #  /_P4_\               .              /_P2_\                    .              .               .
        #  |    |               |              |    |                    .              .               .
        #  T4 -- a --|\    /|-- a --|\    /|-- a -- T2                   |              |               |
        #  |     |   |P3--P3|   |   |P3--P3|   |    |                    C4 -- . . . -- T3 -- . . .  -- C3   
        #  C4 --T3 --|/    \|-- T3--|/    \|-- T3-- C3                  [χ, χ]      [χ, D², χ]       [χ, χ]

        self.T1, self.T2, self.T3, self.T4 = T1, T2, T3, T4
        self.C1, self.C2, self.C3, self.C4 = C1, C2, C3, C4
        
    def _converged(self, tol: float) -> bool:
        return all(abs(sv_sums[-1] - sv_sums[-2]) < tol for sv_sums in [self.sv_sums1, self.sv_sums2, self.sv_sums3, self.sv_sums4])


    def _new_P4(self, R_up: torch.Tensor, R_down: torch.Tensor, grown_chi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the new projector P4.
        """
        R = R_up.view(self.chi, self.D**2, self.chi*self.D**2)
        #   _____                           __|__
        #  |_____|   [χ, D², D², χ]   🡺   |_____|   [χ, D², χD²]
        #  | | | |                         | |   

        R_tilde = R_down.view(self.chi, self.D**2, self.chi*self.D**2)
        #  |_|_|_|                         |_|___
        #  |_____|   [χ, D², D², χ]   🡺   |_____|   [χ, D², χD²]
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

        U, s, Vh = U[:, :grown_chi], s[:grown_chi, :grown_chi], Vh[:grown_chi, :]
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

        # This should decrease as a function of chi. For testing purposes.
        self.diff4 = torch.norm(A - B) / torch.norm(A)

        return P4, P4_tilde
    

    def _new_P2(self, R_up: torch.Tensor, R_down: torch.Tensor, grown_chi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the new projector P2
        """
        R = R_up.view(self.chi*self.D**2, self.D**2, self.chi)
        #   _____                           __|__
        #  |_____|   [χ, D², D², χ]   🡺   |_____|   [χD², D², χ]
        #  | | | |                             | |

        R_tilde = R_down.view(self.chi*self.D**2, self.D**2, self.chi)
        #  |_|_|_|                          ___|_|
        #  |_____|   [χ, D², D², χ]   🡺   |_____|   [χD², D², χ]
        #                                     |

        A = torch.einsum("abc,dbc->ad", R, R_tilde)
        #   __|__  
        #  |_____|
        #   ___|_|    🡺   --o--   [χD², χD²]
        #  |_____|   
        #     |  

        U, s, Vh = torch.linalg.svd(A)
        s = torch.diag(s)
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        U, s, Vh = U[:, :grown_chi], s[:grown_chi, :grown_chi], Vh[:grown_chi, :]
        # --<|---o---|>--   🡺   [χD², χ], [χ, χ], [χ, χD²]

        P2_tilde = torch.einsum("abc,ad,de->cbe", R_tilde, Vh.T, torch.sqrt(torch.linalg.inv(s)))
        P2 = torch.einsum("ab,bc,cde->eda", torch.sqrt(torch.linalg.inv(s)), U.T, R)
        #   ___|_|  
        #  |_____| R~       [χD², D², χ]
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
        #  |_____| R        [χD², D², χ]
        #      | |
        self.sv_sums2.append(torch.sum(s)) 

        return P2, P2_tilde
       
    def _new_P3(self, R_right: torch.Tensor, R_left: torch.Tensor, grown_chi: int) -> torch.Tensor:
        """
        Compute the new projector P3.
        """
        R = R_left.view(self.chi*self.D**2, self.D**2, self.chi)
        #   __ _            __
        #  |  |_           |  |
        #  |  |_    🡺   --|  |_   [χD², D², χ]
        #  |__|_           |__|_

        R_tilde = R_right.view(self.chi*self.D**2, self.D**2, self.chi)
        #  ____            __
        #  _|  |          |  |
        #  _|  |    🡺   _|  |--  [χD², D², χ]
        #  _|__|         _|__|

        A = torch.einsum("abc,dbc->ad", R, R_tilde)
        #     __     __
        #    |  |   |  |
        #  --|  |_ _|  |--   🡺   --o--   [χD², χD²]
        #    |__|_ _|__|

        U, s, Vh = torch.linalg.svd(A)
        s = torch.diag(s)
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        U, s, Vh = U[:, :grown_chi], s[:grown_chi, :grown_chi], Vh[:grown_chi, :]
        # --<|---o---|>--   🡺   [χD², χ], [χ, χ], [χ, χD²]

        P3_tilde = torch.einsum("abc,ad,de->cbe", R_tilde, Vh.T, torch.sqrt(torch.linalg.inv(s)))
        P3 = torch.einsum("ab,bc,cde->eda", torch.sqrt(torch.linalg.inv(s)), U.T, R)
        #     __                                                 __           
        #    |  |        |\                           /|        |  |          --|\           /|--
        #   _|  | ------ | | ---- o -- . . -- o ---- | | ------ |  |_   🡺      | |-- . . --| |
        #   _|__|        |/                           \|        |__|_         --|/           \|--
        #              
        #    R~           V      s^(-1/2)  s^(-1/2)        U†        R          P~            P
        # [χD², D², χ]  [χD², χ]  [χ, χ]   [χ, χ]   [χ, χD²]  [χD², D², χ]    [χ, D², χ]   [χ, D², χ]

        self.sv_sums3.append(torch.sum(s)) 

        return P3, P3_tilde


    def _new_P1(self, R_right: torch.Tensor, R_left: torch.Tensor, grown_chi: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the new projector P1.
        """
        R = R_left.view(self.chi, self.D**2, self.D**2*self.chi)
        #   __ _            ____
        #  |  |_           |  |_
        #  |  |_    🡺   --|  |    [χ, D², D²χ]
        #  |__|_           |__|

        R_tilde = R_right.view(self.chi, self.D**2, self.D**2*self.chi)
        #  ____          ____
        #  _|  |         _|  |
        #  _|  |    🡺    |  |--  [χ, D², D²χ]
        #  _|__|          |__|

        A = torch.einsum("abc,abd->cd", R, R_tilde)
        #     __ ___ __
        #    |  |___|  |
        #  --|  |   |  |--   🡺   --o--   [χD², χD²]
        #    |__|   |__|

        
        U, s, Vh = torch.linalg.svd(A)
        s = torch.diag(s)
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        U, s, Vh = U[:, :grown_chi], s[:grown_chi, :grown_chi], Vh[:grown_chi, :]
        # --<|---o---|>--   🡺   [χD², χ], [χ, χ], [χ, χD²]

        P1_tilde = torch.einsum("abc,cd,de->abe", R_tilde, Vh.T, torch.sqrt(torch.linalg.inv(s)))
        P1 = torch.einsum("ab,bc,dec->dea", torch.sqrt(torch.linalg.inv(s)), U.T, R)
        #   ____                                                 ____           
        #   _|  |        |\                           /|        |  |_         --|\           /|--
        #    |  | ------ | | ---- o -- . . -- o ---- | | ------ |  |    🡺      | |-- . . --| |
        #    |__|        |/                           \|        |__|          --|/           \|--
        #              
        #    R~           V      s^(-1/2)  s^(-1/2)        U†        R          P~            P
        # [χ, D², χD²]  [χD², χ]  [χ, χ]   [χ, χ]   [χ, χD²]  [χ, D², χD²]    [χ, D², χ]   [χ, D², χ]

        self.sv_sums1.append(torch.sum(s)) 

        return P1, P1_tilde
