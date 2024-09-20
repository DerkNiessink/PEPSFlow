import cProfile
import sys
import subprocess

sys.path.append("../..")
from pepsflow.iPEPS_trainer import iPEPSTrainer


with cProfile.Profile() as pr:
    chi = 8
    trainer = iPEPSTrainer(chi=chi, d=2, gpu=False)
    trainer.exe(epochs=5, use_prev=False, runs=1, max_iter=5)

pr.dump_stats("profile.prof")
