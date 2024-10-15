import configparser
import multiprocessing as mp
import os

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


def optimize_single_run(lam: float, args: dict):
    """
    Optimize for a single value of λ.

    Args:
        lam (float): The value of λ.
        args (dict): Dictionary containing the arguments for training the iPEPS model.
    """

    # Set the value of λ and the data file name
    args["lam"] = lam
    if args["data_folder"] != "None":
        args["data_fn"] = os.path.join("data", args["data_folder"], f"lam_{lam}.pth")
    else:
        args["data_fn"] = None

    # Start the training process
    trainer = iPEPSTrainer(args)
    trainer.exe()
    trainer.save_data(os.path.join("data", args["save_folder"], f"lam_{lam}.pth"))


def optimize():

    # Read the parameters from the configuration file
    parser = configparser.ConfigParser()
    parser.read("src/pepsflow/optimize.cfg")
    args = dict(parser["PARAMETERS"])

    # Convert the parameters to the correct data type
    lambdas = eval(args["lam"])
    for key, value in args.items():
        if key not in ("save_folder", "data_folder"):
            args[key] = eval(value)

    # Start multiprocessing for each value of λ
    with mp.Pool(processes=len(lambdas)) as pool:
        pool.starmap(optimize_single_run, [(lam, args.copy()) for lam in lambdas])


if __name__ == "__main__":
    optimize()
