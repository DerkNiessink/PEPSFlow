import pytest

from pepsflow.ipeps.tools import Tools
from pepsflow.ipeps.ipeps import make_ipeps
from pepsflow.ipeps.observe import Observer


class TestOptimization:

    @pytest.mark.parametrize(
        "E_exp, epochs, l, J2, sp, dev, m, D, dt, chi, N",
        [
            (-1.06283, 10, 1, None, False, "cpu", "Ising", 2, "double", 4, 1),
            (-2.06688, 15, 4, None, True, "cuda", "Ising", 2, "single", 4, 1),
            (-0.58900, 20, None, 0.2, True, "cpu", "J1J2", 3, "double", 6, 2),
        ],
    )
    def test_symmetric_optimization(self, E_exp, epochs, l, J2, sp, dev, m, D, dt, chi, N):
        ipeps_args = dict(
            model=m,
            initial_state_symmetry="rotational",
            ctm_symmetry="rotational",
            D=D,
            dtype=dt,
            device=dev,
            lam=l,
            J2=J2,
            split=sp,
            seed=1,
            chi=chi,
            gauge=None,
            projector_mode="eig",
        )
        minimize_args = dict(
            optimizer="lbfgs",
            learning_rate=1,
            epochs=epochs,
            threads=1,
            line_search=True,
            warmup_steps=2,
            gradient_steps=N,
        )
        ipeps = make_ipeps(ipeps_args)
        Tools.minimize(ipeps, minimize_args)
        observer = Observer(ipeps)
        assert observer.optimization_energy() == pytest.approx(E_exp, abs=1e-3)

    def test_general_optimization(self):
        ipeps_args = dict(
            model="J1J2",
            initial_state_symmetry="rotational",
            ctm_symmetry=None,
            D=3,
            dtype="double",
            device="cpu",
            lam=1,
            J2=0.5,
            split=False,
            seed=2,
            chi=8,
            gauge=None,
        )
        minimize_args = dict(
            optimizer="lbfgs",
            learning_rate=1,
            epochs=40,
            threads=1,
            line_search=True,
            warmup_steps=2,
            gradient_steps=5,
        )
        ipeps = make_ipeps(ipeps_args)
        Tools.minimize(ipeps, minimize_args)
        observer = Observer(ipeps)
        assert observer.optimization_energy() == pytest.approx(-0.49104640975943203, abs=1e-3)
