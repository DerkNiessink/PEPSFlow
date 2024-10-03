import pytest

from pepsflow.iPEPS_trainer import iPEPSTrainer


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp", [(1, -1.06283), (4, -2.06688)])
    def test_exe(self, lam, E_exp):
        trainer = iPEPSTrainer(chi=8, d=2, gpu=False, data_fn=None)
        trainer.exe(
            lambdas=[lam],
            epochs=5,
            use_prev=False,
            perturbation=0.0,
            runs=5,
            lr=1,
            max_iter=30,
        )
        E, _, _ = trainer.data[lam].forward()
        assert float(E.detach().numpy()) == pytest.approx(E_exp, abs=1e-3)
