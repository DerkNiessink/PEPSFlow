from pepsflow.train.iPEPS_trainer import iPEPSTrainer

from torch.profiler import ProfilerActivity, profiler
import torch


def profile(gpu: bool, chi: int, d: int, max_iter: int, epochs: int, fn: str = None):
    """
    Profile the iPEPS model using the PyTorch profiler.
    """
    on_trace_ready = (
        profiler.tensorboard_trace_handler("analysis/profiling/log", worker_name=fn)
        if fn
        else None
    )

    with profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=on_trace_ready,
    ) as prof:
        trainer = iPEPSTrainer(chi=chi, d=d, gpu=gpu)
        trainer.exe(epochs=epochs, max_iter=max_iter)

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
    profile(gpu=False, chi=8, d=2, max_iter=10, epochs=5, fn="CPU")
    # tensorboard --logdir=./analysis/profiling/log
