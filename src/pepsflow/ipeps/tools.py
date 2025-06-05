from pepsflow.ipeps.ipeps import iPEPS, GeneralIPEPS
from pepsflow.models.optimizers import Optimizer

import torch
import sys


class Tools:
    """
    A utility class providing static methods for operating on iPEPS models.

    This class includes core functionality for:
    - Minimizing the energy of an iPEPS model via gradient-based optimization.
    - Evaluating the energy of a converged iPEPS state using the CTMRG algorithm.
    - Applying gauge transformations to bring the tensors into canonical or other specified forms.

    All methods are static and operate directly on `iPEPS` instances using configuration
    dictionaries typically parsed from a config file.
    """

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
            tensors = ipeps.do_warmup_steps(N=args["warmup_steps"])
            tensors = ipeps.do_gradient_steps(N=args["gradient_steps"], tensors=tensors)
            loss = ipeps.get_E(grad=True, tensors=tensors)
            loss.backward()
            return loss

        loss = 0
        energies, norms = [], [ipeps.minimal_canonical_criterion() if ipeps.map is None else None]
        for epoch in range(args["epochs"]):
            if args["gauge"] == "minimal_canonical" and ipeps.map is None:
                regauge = (
                    ipeps.minimal_canonical_criterion() > args["gauge_criterion"] or epoch % args["regauge_every"] == 0
                )
            elif args["gauge"] == "simple_update" and ipeps.map is None:
                regauge = epoch % args["regauge_every"] == 0 or epoch == 0
            else:
                regauge = False

            if regauge:
                ipeps.gauge_transform(which=args["gauge"], tolerance=args["gauge_tolerance"])
                ipeps.gauge_transform(which="minimal_canonical", tolerance=args["gauge_tolerance"])
                print(f"Gauge transformed with {args['gauge']} gauge.")

            new_loss: torch.Tensor = opt.step(train)
            norm = ipeps.minimal_canonical_criterion() if ipeps.map is None else None
            sys.stdout.flush()
            print(f"epoch, E, Diff, norm: {epoch, new_loss.item(), abs(new_loss - loss).item(), norm}")
            energies.append(new_loss.item())
            norms.append(norm)

            if abs(new_loss - loss) < args.get("tolerance", 1e-10):
                sys.stdout.flush()
                print(f"Converged after {epoch} epochs. Saving and quiting training...")
                break
            loss = new_loss

        ipeps.add_data("optimization", {"args": args, "energies": energies, "norms": norms})

    @staticmethod
    def evaluate(ipeps: iPEPS, args: dict) -> None:
        """
        Compute the energy of a converged iPEPS state for a given bond dimension using the CTMRG
        algorithm.

        Args:
            ipeps (iPEPS): iPEPS model to compute the energies for.
            args (dict): Dictionary containing the iPEPS parameters.
        """
        data = {"args": args, "energies": []}

        for chi in args["chi"]:
            tensors = ipeps.do_evaluation(
                N=args["ctm_steps"], chi=chi, ctm_symmetry=args["ctm_symmetry"], projector_mode=args["projector_mode"]
            )
            E = ipeps.get_E(grad=False, tensors=tensors)
            data["energies"].append(E.item())
            print(f"chi, E: {chi, E.item()}")

        ipeps.add_data("evaluation", data)

    @staticmethod
    def gauge(ipeps: GeneralIPEPS, args: dict) -> None:
        """Gauge transform the iPEPS tensor.

        Args:
            ipeps (iPEPS): iPEPS model to optimize.
            args (dict): Dictionary containing the arguments for the optimization process.
        """
        ipeps.set_seed(args["seed"])
        if ipeps.map is not None:
            raise ValueError("The given iPEPS is rotationally symmetric. No gauge transformation is needed.")
        ipeps.gauge_transform(which=args["gauge"], tolerance=args["tolerance"])
        ipeps.add_data("gauge", args)
