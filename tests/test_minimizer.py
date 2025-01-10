import pytest

from pepsflow.ipeps.minimizer import Minimizer
from pepsflow.ipeps.ipeps import iPEPS


class TestMinimizer:

    @pytest.mark.parametrize(
        "E_exp, epochs, l, J2, sp, dev, m, D, dt, chi, N",
        [
            (-1.06283, 15, 1, None, False, "cpu", "Ising", 2, "double", 4, 5),
            (-2.06688, 20, 4, None, True, "cuda", "Ising", 2, "single", 4, 5),
            (-0.58900, 25, None, 0.2, True, "cpu", "J1J2", 3, "double", 16, 10),
        ],
    )
    def test_trainer(self, E_exp, epochs, l, J2, sp, dev, m, D, dt, chi, N):
        ipeps_args = dict(
            model=m,
            D=D,
            dtype=dt,
            device=dev,
            lam=l,
            J2=J2,
            split=sp,
            seed=1,
            chi=chi,
            warmup_steps=1,
            Niter=N,
            save_intermediate=False,
        )
        minimize_args = dict(optimizer="lbfgs", learning_rate=1, epochs=epochs, threads=1, line_search=True, log=False)
        minimizer = Minimizer(iPEPS(ipeps_args), minimize_args)

        minimizer.minimize()
        assert minimizer.ipeps.data["losses"][-1].cpu().detach() == pytest.approx(E_exp, abs=1e-3)
