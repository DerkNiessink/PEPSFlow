import pytest
import matplotlib.pyplot as plt

from pepsflow.ipeps.io import IO
from pepsflow.ipeps.ipeps import make_ipeps
from pepsflow.ipeps.tools import Tools


class TestGradientConvergence:

    def test_gradient_convergence(self):
        """
        Test the convergence of the gradient during optimization.
        The gradient should converge to zero within a small tolerance.
        """
        grad_norms = []
        for nsteps in range(35):
            ipeps_args = dict(
                model="J1J2",
                initial_state_symmetry="rotational",
                ctm_symmetry="rotational",
                D=2,
                dtype="double",
                device="cpu",
                lam=1,
                J2=0,
                split=True,
                seed=5,
                chi=4,
                noise=0,
                gauge=None,
                projector_mode="eig",
            )
            minimize_args = dict(
                optimizer="lbfgs",
                learning_rate=1,
                epochs=1,
                threads=1,
                line_search=True,
                warmup_steps=0,
                gradient_steps=nsteps,
                gauge=None,
            )
            ipeps = make_ipeps(ipeps_args)
            Tools.minimize(ipeps, minimize_args)
            grad_norm = 0.0

            for param in ipeps.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norms.append(grad_norm**0.5)

        grad_diffs = [abs(grad_norms[i + 1] - grad_norms[i]) for i in range(len(grad_norms) - 1)]
        print(grad_diffs)
        plt.plot(range(1, 35), grad_diffs, marker="o")
        plt.xlabel("Gradient Steps")
        plt.ylabel("Consecutive Gradient Norm Difference")
        plt.yscale("log")
        plt.show()
