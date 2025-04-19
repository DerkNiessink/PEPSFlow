import pytest

from pepsflow.iPEPS.tools import Tools
from pepsflow.iPEPS.iPEPS import make_ipeps


class TestMinimizer:

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
            rotational_symmetry="both",
            D=D,
            dtype=dt,
            device=dev,
            lam=l,
            J2=J2,
            split=sp,
            seed=1,
            chi=chi,
            warmup_steps=2,
            Niter=N,
        )
        minimize_args = dict(optimizer="lbfgs", learning_rate=1, epochs=epochs, threads=1, line_search=True, log=False)
        ipeps = make_ipeps(ipeps_args)
        Tools.minimize(ipeps, minimize_args)

        assert ipeps.data["losses"][-1] == pytest.approx(E_exp, abs=1e-3)


