import pytest

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp", [(1, -1.06283), (4, -2.06688)])
    def test_exe(self, lam, E_exp):
        args = {
            "chi": 8,
            "D": 2,
            "data_fn": None,
            "gpu": False,
            "lam": lam,
            "max_iter": 30,
            "runs": 5,
            "lr": 1,
            "epochs": 5,
            "use_prev": False,
            "perturbation": 0.0,
        }
        trainer = iPEPSTrainer(args)
        trainer.exe()
        E, _, _ = trainer.data.forward()
        assert float(E.detach().numpy()) == pytest.approx(E_exp, abs=1e-3)
