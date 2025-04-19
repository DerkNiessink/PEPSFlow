from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import sys


class Tools:

    @staticmethod
    def minimize(ipeps: iPEPS, args: dict):
        """
        Minimize the energy of the iPEPS model using automatic differentiation.

        Args:
            ipeps (iPEPS): iPEPS model to optimize.
            args (dict): Dictionary containing the arguments for the optimization process.
        """

        torch.set_num_threads(args["threads"])
        ls = "strong_wolfe" if args["line_search"] else None
        opt = Optimizer(args["optimizer"], ipeps.parameters(), lr=args["learning_rate"], line_search_fn=ls)

        def train() -> torch.Tensor:
            opt.zero_grad()
            tensors = ipeps.do_warmup_steps()
            tensors = ipeps.do_gradient_steps(tensors=tensors)
            loss = ipeps.get_E(grad=True, tensors=tensors)
            loss.backward()
            return loss

        loss = 0
        for epoch in range(args["epochs"]):

            new_loss: torch.Tensor = opt.step(train)
            sys.stdout.flush()
            print(f"epoch, E, Diff: {epoch, new_loss.item(), abs(new_loss - loss).item()}")
            ipeps.add_data(new_loss.item())

            if abs(new_loss - loss) < 1e-15:
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
        ipeps.plant_gauge()
        ipeps.args["chi"] = args["chi"]
        with torch.no_grad():
            tensors = ipeps.do_evaluation()
        E = ipeps.get_E(grad=False, tensors=tensors)
        ipeps.add_data(E.item())
        print(f"chi, E: {ipeps.args['chi'], E.item()}")
