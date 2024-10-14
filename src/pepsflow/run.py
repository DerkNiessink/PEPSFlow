import argparse

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


# fmt: off

parser = argparse.ArgumentParser(prog="pepsflow", description="Optimize the iPEPS tensor for the specified parameters")

parser.add_argument("--chi", type=int, default=6, help="Environment bond dimension in the CTMRG algorithm")
parser.add_argument("--D", type=int, default=2, help="Bulk bond dimension of the iPEPS")
parser.add_argument("--data_fn", type=str, default=None, help="Filename of data with extension .pth containing an iPEPS model")
parser.add_argument("--gpu", action="store_true", help="Run the model on the GPU if available")
parser.add_argument("--no-gpu", action="store_false", help="Run the model on the CPU")
parser.add_argument("--lam", type=float, default=0.5, help="Value of the parameter lambda in the tranverse-field Ising model")
parser.add_argument("--max_iter", type=int, default=20, help="Maximum number of iterations for the optimizer")
parser.add_argument("--runs", type=int, default=1, help="Number of runs to train the model. Applies to random initialization. The program will choose the best model based on the lowest energy.")
parser.add_argument("--lr", type=float, default=1, help="Learning rate for the optimizer")
parser.add_argument("--epochs", type=int, default=10, help="Maximum number of epochs to train the model")
parser.add_argument("--perturbation", type=float, default=0.0, help="Amount of perturbation to apply to the initial state")
parser.add_argument("--save_fn", type=str, default=None, help="Filename to save the iPEPS model in.")
parser.add_argument("--threads", type=int, default=8, help="Number of threads to use for the optimization. Each thread runs on a separate CPU core.")

args = vars(parser.parse_args())
trainer = iPEPSTrainer(args)
trainer.exe()
trainer.save_data(args["save_fn"])
