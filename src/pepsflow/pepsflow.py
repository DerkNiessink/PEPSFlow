import configparser
import multiprocessing as mp
import os
import ast

from pepsflow.iPEPS.trainer import Trainer
from pepsflow.iPEPS.converger import Converger


def path(folder: str, file: str) -> str:
    """
    Return the path to the file in the folder.

    Args:
        folder (str): Folder containing the file.
        file (str): Filename.

    Returns:
        str: Path to the file.
    """
    return os.path.join("data", folder, file)


def read_config() -> dict:
    """
    Read the parameters from the configuration file.

    Returns:
        dict: Dictionary containing the parameters.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str  # Preserve the case of the keys
    parser.read("src/pepsflow/pepsflow.cfg")
    args = dict(parser["PARAMETERS"])
    args["var_param"] = None

    for key, value in args.items():

        try:
            args[key] = ast.literal_eval(value)
            args[key] = None if value == "None" else args[key]
            args["var_param"] = key if type(args[key]) == list else args["var_param"]

        except ValueError:
            pass

    if args["var_param"] is None:
        raise ValueError("No variational parameter found.")

    return args


def optimize(value: float, args: dict):
    """
    Optimize the iPEPS model for a single value the variational parameter.

    Args:
        value (float): Value of the variational parameter.
        param (str): Name of the variational parameter.
        args (dict): Arguments for the optimization.
    """
    # Set the value of the variational parameter
    args[args["var_param"]] = value
    fn = f"{args['var_param']}_{value}.pth"

    trainer = Trainer(args)
    if args["read_folder"]:
        trainer.read(path(args["read_folder"], fn))
    trainer.exe()
    trainer.write(path(args["write_folder"], fn))


def converge(value: float, args: dict, read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
    algorithm.

    Args:
        value (float): Value of the variational parameter.
        args (dict): Arguments for the optimization.
        read_fn (str): Filename of the data file to read from
    """
    # Set the value of the variational parameter
    if args["var_param"] == "chi":
        args[args["var_param"]] = value
        fn = f"{args['var_param']}_{value}.pth"
    else:
        raise ValueError("Only chi as variational parameter is supported for convergence.")

    conv = Converger(args)
    conv.read(path(args["read_folder"], read_fn))
    conv.exe()
    conv.write(path(args["write_folder"], fn))


def optimize_parallel():
    """
    Optimize the iPEPS model for a list of values of the variational parameter.
    """
    args = read_config()
    with mp.Pool(processes=len(args[args["var_param"]])) as pool:
        pool.starmap(optimize, [(value, args.copy()) for value in args[args["var_param"]]])


def converge_parallel(read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a list of bond dimensions using the CTMRG
    algorithm.

    Args:
        read_fn (str): Filename of the data file to read from.
    """
    args = read_config()
    with mp.Pool(processes=len(args[args["var_param"]])) as pool:
        pool.starmap(converge, [(value, args.copy(), read_fn) for value in args[args["var_param"]]])
