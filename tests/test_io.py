import pytest

from pepsflow.ipeps.tools import Tools
from pepsflow.ipeps.io import IO
from pepsflow.ipeps.ipeps import make_ipeps


class TestMinimizer:

    def test_save(self):
        ipeps_args = dict(
            model="J1J2",
            rotational_symmetry="both",
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

        # Update ipeps args and minimize args
        ipeps_args2 = ipeps_args.copy()
        ipeps_args2["chi"] = 12
        ipeps_args2["rotational_symmetry"] = None
        minimize_args2 = minimize_args.copy()
        minimize_args2["epochs"] = 10

        ipeps = IO.load("tests/test_data/test")
        ipeps2 = make_ipeps(ipeps_args2, initial_ipeps=ipeps)
        Tools.minimize(ipeps2, minimize_args2)
        IO.save(ipeps2, "tests/test_data/test")
        assert (ipeps2.data["energies"][-1] == pytest.approx(-0.6602310934799586, abs=1e-3)) and (
            len(ipeps2.data["energies"]) == 13
        )
