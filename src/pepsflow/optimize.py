import configparser
import multiprocessing as mp
import os
import ast

from pepsflow.train.iPEPS_trainer import iPEPSTrainer


def optimize_single_run(value: float, param, args: dict):
    """
    Optimize the iPEPS model for a single value the variational parameter.

    Args:
        value (float): Value of the variational parameter.
        param (str): Name of the variational parameter.
        args (dict): Arguments for the optimization.
    """

    # Set the value of the variational
    args[param] = value
    args["var_param"] = param
    fn = f"{param}_{value}.pth"

    # Set the data file name
    if args["data_folder"] != "None":
        args["data_fn"] = os.path.join("data", args["data_folder"], fn)
    else:
        args["data_fn"] = None

    # Start the training process
    trainer = iPEPSTrainer(args)
    trainer.exe()
    trainer.save_data(os.path.join("data", args["save_folder"], fn))


def optimize():

    # Read the parameters from the configuration file
    parser = configparser.ConfigParser()
    parser.read("src/pepsflow/optimize.cfg")
    args = dict(parser["PARAMETERS"])

    var_param = None
    for key, value in args.items():

        # Skip the save_folder, data_folder, and model parameters.
        # These already have the correct data type (str).
        if key not in ("save_folder", "data_folder", "model"):
            args[key] = ast.literal_eval(value)
            # Find the variational parameter
            if type(args[key]) == list:
                var_param = key

    if var_param is None:
        raise ValueError("No variational parameter found.")

    # Start multiprocessing for each value of λ
    with mp.Pool(processes=len(args[var_param])) as pool:
        pool.starmap(
            optimize_single_run,
            [(value, var_param, args.copy()) for value in args[var_param]],
        )


if __name__ == "__main__":
    optimize()
