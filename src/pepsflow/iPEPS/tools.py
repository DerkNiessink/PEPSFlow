from pepsflow.ipeps.ipeps import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import sys
import matplotlib.pyplot as plt
from scipy.linalg import expm


class Tools:

    @staticmethod
    def minimize(ipeps: iPEPS, args: dict):
        """
        Minimize the energy of the iPEPS model using automatic differentiation.

        Args:
            ipeps (iPEPS): iPEPS model to optimize.
            args (dict): Dictionary containing the arguments for the optimization process.
        """
        ipeps.do_gauge_transform()
        ipeps.add_data(key="Gauges [U1, U2]", value=ipeps.U1)
        ipeps.add_data(key="Gauges [U1, U2]", value=ipeps.U2)

        torch.set_num_threads(args["threads"])
        ls = "strong_wolfe" if args["line_search"] else None
        opt = Optimizer(args["optimizer"], ipeps.parameters(), lr=args["learning_rate"], line_search_fn=ls)

        def train() -> torch.Tensor:
            opt.zero_grad()
            tensors = ipeps.do_warmup_steps()
            tensors = ipeps.do_gradient_steps(tensors=tensors)
            loss = ipeps.get_E(grad=True, tensors=tensors)
            loss.backward()

            for p in ipeps.parameters():
                if p.grad is not None and not p.grad.is_contiguous():
                    p.grad = p.grad.contiguous()
            return loss

        loss = 0
        for epoch in range(args["epochs"]):

            new_loss: torch.Tensor = opt.step(train)
            sys.stdout.flush()
            print(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")
            ipeps.add_data(key="energies", value=new_loss.item())

            if abs(new_loss - loss) < args.get("tol", 1e-10):
                sys.stdout.flush()
                print(f"Converged after {epoch} epochs. Saving and quiting training...")
                break
            loss = new_loss

    @staticmethod
    def evaluate(ipeps: iPEPS, args: dict) -> None:
        """
        Compute the energy of a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm.

        Args:
            ipeps (iPEPS): iPEPS model to compute the energies for.
            args (dict): Dictionary containing the iPEPS parameters.
        """
        ipeps.args = args
        ipeps.do_gauge_transform()
        ipeps.add_data(key="Gauges [U1, U2]", value=ipeps.U1)
        ipeps.add_data(key="Gauges [U1, U2]", value=ipeps.U2)
        with torch.no_grad():
            tensors = ipeps.do_evaluation()
        E = ipeps.get_E(grad=False, tensors=tensors)
        ipeps.add_data(key="Eval_energy", value=E.item())
        print(f"chi, E: {ipeps.args['chi'], E.item()}")

    @staticmethod
    def minimize_norm(ipeps: iPEPS, args: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the minimal canonical form of the iPEPS tensor according to the algorithm 1 in
        https://arxiv.org/pdf/2209.14358v1.

        Args:
            ipeps (iPEPS): iPEPS model to optimize.
            args (dict): Dictionary containing the arguments for the optimization process.

        Returns:
            Set of gauge transformations (g1, g2) that minimize the norm of the iPEPS tensor.

        """
        A = ipeps.params[ipeps.map] if ipeps.map is not None else ipeps.params
        g1 = g2 = ipeps.tensors.identity(ipeps.args["D"])
        y = []
        while True:

            g1_inv = torch.linalg.inv(g1)
            g2_inv = torch.linalg.inv(g2)
            A = torch.einsum("purdl,Uu,Rr,dD,lL->pURDL", A, g2, g1, g2_inv, g1_inv)
            #            g2^-1
            #            /                      /
            #    g1  -- o -- g1^-1       🡺 -- o --
            #          /|                     /|
            #       g2

            rho = torch.einsum("purdl,pURDL->urdlURDL", A, A)
            #      /
            #  -- o --
            #    /|/      🡺   [D, D]
            #  -- o --
            #    /

            rho_11 = torch.einsum("urdluRdl->rR", rho)
            rho_12 = torch.einsum("urdlurdL->lL", rho)
            # tracing over all legs except the right and left legs respectively:
            #      /|         /|
            #  -- o---    -- o---
            #  | /|/    ,   /|/ |   🡺   [D, D], [D, D]
            #  -|-o --    -|-o --
            #   |/         |/

            rho_21 = torch.einsum("urdlUrdl->uU", rho)
            rho_22 = torch.einsum("urdlurDl->dD", rho)
            # tracing over all legs except the up and down legs respectively:
            #      /           /|
            #  -- o --     -- o---
            #  | /|/ |  ,  | /|/ |   🡺   [D, D], [D, D]
            #  -|-o --     -- o --
            #   |/           /

            trace_rho = torch.einsum("urdlurdl->", rho)
            # tracing over all legs:
            #      /|
            #  -- o---
            #  | /|/ |
            #  -|-o --
            #   |/

            diff1 = rho_11 - rho_12.T
            diff2 = rho_21 - rho_22.T

            f = (1 / trace_rho) * (diff1.norm() ** 2 + diff2.norm() ** 2)
            y.append(f.item())
            if f < args["tolerance"]:
                break

            g1 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff1)
            g2 = torch.linalg.matrix_exp((-1 / (8 * trace_rho)) * diff2)

        ipeps.add_data(key="Gauges [U1, U2]", value=g1)
        ipeps.add_data(key="Gauges [U1, U2]", value=g2)

        plt.plot(range(len(y)), y, marker="o", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("Norm")
        plt.show()
