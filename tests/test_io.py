import pytest

from pepsflow.ipeps.tools import Tools
from pepsflow.ipeps.io import IO
from pepsflow.ipeps.ipeps import make_ipeps
from pepsflow.ipeps.observe import Observer


class TestMinimizer:

    def test_save(self):
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
            optimizer="lbfgs", learning_rate=1, epochs=3, threads=1, line_search=True, warmup_steps=2, gradient_steps=4
        )
        ipeps = make_ipeps(ipeps_args)
        Tools.minimize(ipeps, minimize_args)
        IO.save(ipeps, "tests/test_data/test")

        # Update ipeps args and opt args
        ipeps_args2 = ipeps_args.copy()
        ipeps_args2["chi"] = 12
        ipeps_args2["ctm_symmetry"] = None
        opt_args2 = minimize_args.copy()
        opt_args2["epochs"] = 10

        ipeps = IO.load("tests/test_data/test")
        ipeps2 = make_ipeps(ipeps_args2, initial_ipeps=ipeps)
        Tools.minimize(ipeps2, opt_args2)
        IO.save(ipeps2, "tests/test_data/test")
        observer = Observer(ipeps2)

        total_len = len(observer.optimization_energies(0)) + len(observer.optimization_energies(1))
        assert total_len == 13
        assert observer.optimization_energy() == pytest.approx(-0.6602310934799586, abs=1e-3)
