import pytest

from pepsflow.iPEPS.trainer import Trainer
from pepsflow.iPEPS.iPEPS import iPEPS


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp, split, device", [(1, -1.06283, False, "cpu"), (4, -2.06688, True, "cuda")])
    def test_Ising(self, lam, E_exp, split, device):

        ipeps = iPEPS(
            {
                "model": "Ising",
                "D": 2,
                "lam": lam,
                "split": split,
                "chi": 4,
                "Niter": 5,
                "warmup_steps": 5,
                "seed": 1,
                "dtype": "single",
                "device": device,
            }
        )
        trainer = Trainer(
            ipeps,
            {
                "optimizer": "lbfgs",
                "learning_rate": 1,
                "epochs": 20,
                "threads": 1,
                "line_search": True,
                "log": False,
            },
        )

        trainer.exe()
        assert ipeps.data["losses"][-1].cpu().detach() == pytest.approx(E_exp, abs=1e-4)

    def test_J1J2(self):
        ipeps = iPEPS(
            {
                "model": "Ising",
                "D": 2,
                "lam": 1,
                "split": True,
                "chi": 4,
                "Niter": 5,
                "warmup_steps": 5,
                "seed": 1,
                "dtype": "single",
                "device": "cpu",
            }
        )
        trainer = Trainer(
            ipeps,
            {
                "optimizer": "lbfgs",
                "learning_rate": 1,
                "epochs": 20,
                "threads": 1,
                "line_search": True,
                "log": False,
            },
        )
