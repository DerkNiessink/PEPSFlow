import time
import torch
import torch.backends.opt_einsum
from torch.nn.functional import normalize
from rich.progress import Progress, Task

from pepsflow.models.tensors import Methods
from pepsflow.models.svd import CustomSVD

norm = Methods.normalize
symm = Methods.symmetrize


class CtmAlg:
    """
    Class for the Corner Transfer Matrix (CTM) algorithm.

    Args:
        A (torch.Tensor): initial tensor to insert in the CTM algorithm
        chi (int): bond dimension of the edge and corner tensors. Default is 2.
        C (torch.Tensor): initial corner tensor for the CTM algorithm. Default is None
        T (torch.Tensor): initial edge tensor for the CTM algorithm. Default is None
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

    def __init__(self, A: torch.Tensor, chi: int, C: torch.Tensor = None, T: torch.Tensor = None, split: bool = False):
        D = A.size(1)
        self.D = D
        self.split = split
        self.max_chi = chi
        self.chi = D**2 if C is None else C.size(0)  # In both cases we have to let chi grow to max_chi.
        self.trunc_errors = []

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

    def exe(self, N: int = 1, progress: Progress = None, task: Task = None):
        """
        Execute the CTM algorithm for N steps.

        `N` (int): number of steps to execute the algorithm for.
        `progress` (rich.progress.Progress): Progress bar to visualize the progress of the algorithm. Default is None.
        `task` (rich.progress.Task): Task to update the progress bar with. Default is None.
        """
        start = time.time()
        progress.start_task(task) if progress else None
        for _ in range(N):
            self.split_step() if self.split else self.classic_step()
            progress.update(task, advance=1) if progress else None
        self.exe_time = time.time() - start
        if self.split:
            self.T = self.T.view(self.chi, self.D**2, self.chi)

    def classic_step(self):
        """
        Execute one "classic" CTM step. This is the standard CTM algorithm for the rank-4 input tensor.
        """
        # fmt: off
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

    def split_step(self):
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

        U, s, Vh = CustomSVD.apply(M)
        #  --o--   🡺   --<|---o---|>--  [χD², χD²], [χD², χD²], [χD², χD²]

        self.trunc_errors.append(torch.sum(normalize(s, p=1, dim=0)[self.chi :]).float())

        # Let chi grow if the desired chi is not yet reached.
        k = self.chi
        self.chi = min(self.chi * self.D**2, self.max_chi)
        U = U[:, : self.chi]

        # Reshape U back in a rank-3 or 4 tensor.
        shape = (k, self.D, self.D, self.chi) if self.split else (k, self.D**2, self.chi)
        return U.view(shape)
