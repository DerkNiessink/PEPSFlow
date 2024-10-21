import pytest

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp", [(1, -1.06283), (4, -2.06688)])
    def test_exe(self, lam, E_exp):
        args = {
            "model": "Ising",
            "chi": 8,
            "d": 2,
            "data_fn": None,
            "gpu": False,
            "lam": lam,
            "runs": 5,
            "learning_rate": 1,
            "epochs": 20,
            "use_prev": False,
            "perturbation": 0.0,
            "threads": 1,
            "var_param": "lam",
        }
        trainer = iPEPSTrainer(args)
        trainer.exe()
        E, _, _ = trainer.data.forward()
        assert float(E.detach().numpy()) == pytest.approx(E_exp, abs=1e-3)
