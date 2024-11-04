import pytest

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp, split", [(1, -1.06283, False), (4, -2.06688, True)])
    def test_exe(self, lam, E_exp, split):
        args = {
            "model": "Ising",
            "chi": 8,
            "d": 2,
            "split": split,
            "data_fn": None,
            "gpu": False,
            "lam": lam,
            "runs": 3,
            "learning_rate": 0.5,
            "epochs": 20,
            "warmup_steps": 200,
            "perturbation": 0.0,
            "threads": 1,
            "var_param": "lam",
            "line_search": False,
        }
        trainer = iPEPSTrainer(args)
        trainer.exe()
        E = trainer.data.losses[-1]
        assert E == pytest.approx(E_exp, abs=1e-3)
