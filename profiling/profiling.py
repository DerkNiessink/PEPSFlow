from pepsflow.iPEPS.minimizer import Trainer
from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.iPEPS.reader import iPEPSReader

from torch.profiler import ProfilerActivity, profiler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn


progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
)


def profile(fn: str = None):
    """
    Profile the iPEPS model using the PyTorch profiler.
    """
    on_trace_ready = profiler.tensorboard_trace_handler("profiling/log", worker_name=fn) if fn else None

    start_ipeps = iPEPSReader("data/profiling/start_state_D_4.pth").iPEPS
    ipeps = iPEPS(
        {
            "model": "Heisenberg",
            "D": 4,
            "lam": 1,
            "split": True,
            "chi": 40,
            "Niter": 10,
            "warmup_steps": 50,
            "seed": 3,
            "dtype": "single",
            "device": "cuda",
            "start_epoch": -1,
            "perturbation": 0,
        },
        initial_ipeps=start_ipeps,
    )

    with profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=on_trace_ready,
    ) as prof:
        trainer = Trainer(
            ipeps,
            {
                "optimizer": "lbfgs",
                "learning_rate": 1,
                "epochs": 1,
                "threads": 1,
                "line_search": True,
                "log": True,
            },
        )
        task = progress.add_task(f"[blue bold]Training iPEPS", total=trainer.args["epochs"], start=False)
        trainer.exe(task=task, progress=progress)

    return prof


def total_cpu_time(prof: profiler.profile):
    """
    Return the total CPU time from the profiler output in seconds.
    """
    return sum(entry.self_cpu_time_total for entry in prof.key_averages())


def extract_row(prof: profiler.profile, name: str):
    """
    Extract a row from the profiler output.
    """
    for row in prof.key_averages():
        if row.key == name:
            return row


def percentage_cpu_time(prof: profiler.profile, name: str):
    """
    Return the percentage of CPU time for a given operation from the profiler output.
    """
    row = extract_row(prof, name)
    return row.cpu_time_total / total_cpu_time(prof) * 100


if __name__ == "__main__":
    # profile(fn="CPU")
    profile(fn="GPU")

    # tensorboard --logdir=./log
