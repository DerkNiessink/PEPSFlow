import pytest

from pepsflow.iPEPS_trainer import iPEPSTrainer


class TestiPEPSTrainer:

    def test_exe(self):
        trainer = iPEPSTrainer(chi=2, d=2, gpu=False, data_fn=None)
        trainer.exe(
            lambdas=[1],
            epochs=10,
            use_prev=False,
            perturbation=0.0,
            runs=5,
            lr=1,
            max_iter=20,
        )
