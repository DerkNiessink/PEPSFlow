import configparser
import multiprocessing as mp
import os
import ast

from pepsflow.iPEPS.trainer import Trainer
from pepsflow.iPEPS.converger import Converger


def read_config():
    """
    Read the parameters from the configuration file.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = str  # Preserve the case of the keys
    parser.read("src/pepsflow/pepsflow.cfg")
    args = dict(parser["PARAMETERS"])

    var_param = None
    for key, value in args.items():
        if value == "None":
            args[key] = None

        # These already have the correct data type (str).
        if key not in ("write_folder", "read_folder", "model", "optimizer"):

            args[key] = ast.literal_eval(value)
            # Find the variational parameter
            if type(args[key]) == list:
                var_param = key

    if var_param is None:
        raise ValueError("No variational parameter found.")

    return args, var_param


def get_save_path(value: float, param: str, args: dict, fn: str = None) -> str:
    """
    Get the filename for the data file.

    Args:
        value (float): Value of the variational parameter.
        param (str): Name of the variational parameter.
        args (dict): Arguments for the optimization.
        read_fn (str): Filename of the data file to read from and save to.

    Returns:
        str: Filename for the data file.
    """
    # Set the value of the variational
    args[param] = value
    args["var_param"] = param
    fn = f"{param}_{value}" if not fn else fn

    # Set the data file name
    args["data_fn"] = os.path.join("data", args["read_folder"], fn) if args["read_folder"] != None else None

    return os.path.join("data", args["write_folder"], fn)


def optimize_single_run(value: float, param: str, args: dict):
    """
    Optimize the iPEPS model for a single value the variational parameter.

    Args:
        value (float): Value of the variational parameter.
        param (str): Name of the variational parameter.
        args (dict): Arguments for the optimization.
    """
    path = get_save_path(value, param, args)
    trainer = Trainer(args)
    trainer.exe()
    trainer.save_data(path)


def converge_single_run(value: float, param: str, args: dict, read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
    algorithm.

    Args:
        value (float): Value of the variational parameter.
        param (str): Name of the variational parameter.
        args (dict): Arguments for the optimization.
        read_fn (str): Filename of the data file to read from
    """
    args["chi"] = value
    path = get_save_path(value, param, args, read_fn)
    conv = Converger(args)
    conv.exe()
    conv.save_data(path)


def optimize():
    """
    Optimize the iPEPS model for a list of values of the variational parameter.
    """
    args, var_param = read_config()
    with mp.Pool(processes=len(args[var_param])) as pool:
        pool.starmap(
            optimize_single_run,
            [(value, var_param, args.copy()) for value in args[var_param]],
        )


def converge(read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a list of bond dimensions using the CTMRG
    algorithm.

    Args:
        read_fn (str): Filename of the data file to read from.
    """
    args, var_param = read_config()
    with mp.Pool(processes=len(args[var_param])) as pool:
        pool.starmap(
            converge_single_run,
            [(value, var_param, args.copy(), read_fn) for value in args[var_param]],
        )
