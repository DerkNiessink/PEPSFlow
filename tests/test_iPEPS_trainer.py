import pytest
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from pepsflow.iPEPS.trainer import Trainer
from pepsflow.iPEPS.iPEPS import iPEPS

# Global progress bar
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
)


class TestiPEPSTrainer:

    @pytest.mark.parametrize("lam, E_exp, split", [(1, -1.06283, False), (4, -2.06688, True)])
    def test_exe(self, lam, E_exp, split):

        ipeps = iPEPS(
            {
                "model": "Ising",
                "D": 2,
                "lam": lam,
                "split": split,
                "chi": 4,
                "Niter": 10,
                "warmup_steps": 50,
                "seed": 1,
            }
        )
        trainer = Trainer(
            ipeps,
            {
                "optimizer": "lbfgs",
                "learning_rate": 1,
                "epochs": 25,
                "threads": 1,
                "line_search": True,
                "log": True,
            },
        )
        task = progress.add_task(
            f"[blue bold]Training iPEPS ({"lam"} = {lam})", total=trainer.args["epochs"], start=False
        )

        trainer.exe(progress, task)
        assert ipeps.data["losses"][-1].detach() == pytest.approx(E_exp, abs=1e-4)
