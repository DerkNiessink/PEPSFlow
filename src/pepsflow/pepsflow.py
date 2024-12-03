import configparser
import multiprocessing as mp
import os
import ast
from rich.progress import Progress, TextColumn, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn, BarColumn

from pepsflow.iPEPS.trainer import Trainer
from pepsflow.iPEPS.converger import Converger
from pepsflow.iPEPS.iPEPS import iPEPS
from pepsflow.iPEPS.reader import iPEPSReader

progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
)


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


def read_config() -> tuple[dict, tuple[str, str]]:
    """
    Read the parameters from the configuration file.

    Returns:
        dict: Dictionary containing the parameters.
        tuple: Tuple containing the section and key of the varying parameter.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = lambda option: option  # Preserve the case of the keys
    parser.read("src/pepsflow/pepsflow.cfg")

    var_param = None
    args = {section: {} for section in parser.sections()}
    for section in args.keys():
        for key, value in parser.items(section):
            try:
                args[section][key] = ast.literal_eval(value)

                if isinstance(args[section][key], list):
                    if var_param:
                        raise KeyError("Only one varying parameter is allowed.")
                    var_param = (section, key)

            # If the value is no type and should be a string
            except ValueError:
                pass

    if var_param is None:
        raise KeyError("No variational parameter found.")

    return args, var_param


def optimize(var_param: tuple[str, str], value: float, args: dict):
    """
    Optimize the iPEPS model for a single value the variational parameter.

    Args:
        var_param (tuple): Section and key of the variational parameter.
        value (float): Value of the variational parameter.
        args (dict): folder, ipeps, and optimization parameters.
    """
    task = progress.add_task(f"[blue bold]Training iPEPS ({key} = {value})", total=opt_params["epochs"], start=False)

    # Set the value of the variational parameter
    section, key = var_param
    args[section][key] = value
    fn = f"{key}_{value}"

    folders = args["parameters.folders"]
    ipeps_params = args["parameters.ipeps"]
    opt_params = args["parameters.optimization"]

    # Read the iPEPS model from a file if specified
    if folders["read"]:
        ipeps = iPEPSReader(path(folders["read"], fn)).iPEPS
        ipeps = iPEPS(args=ipeps_params, initial_ipeps=ipeps)
    else:
        ipeps = iPEPS(ipeps_params)

    # Execute the optimization and write the iPEPS model to a file
    trainer = Trainer(ipeps, opt_params)
    trainer.exe(progress, task)
    trainer.write(path(folders["write"], fn))


def converge(var_param, value: float, args: dict, read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a given bond dimension using the CTMRG
    algorithm.

    Args:
        value (float): Value of the variational parameter.
        args (dict): Arguments for the optimization.
        read_fn (str): Filename of the data file to read from
    """

    # Set the value of the variational parameter
    section, key = var_param
    if key == "chi":
        args[section][key] = value
        write_fn = f"{key}_{value}"
    else:
        raise KeyError("Only chi as variational parameter is supported for convergence.")

    folders, ipeps_params = args["parameters.folders"], args["parameters.ipeps"]

    ipeps = iPEPSReader(path(folders["read"], read_fn)).iPEPS

    # Execute the convergence and write the data to a file
    conv = Converger(ipeps, ipeps_params)
    conv.exe()
    conv.write(path(folders["write"], write_fn))


def optimize_parallel():
    """
    Optimize the iPEPS model for a list of values of the variational parameter.
    """
    args, var_param = read_config()
    section, key = var_param
    var_param_values = args[section][key]
    num_processes = len(var_param_values)

    with mp.Pool(num_processes) as pool:
        pool.starmap(optimize, [(var_param, value, args.copy()) for value in var_param_values])


def converge_parallel(read_fn: str):
    """
    Compute the energy if a converged iPEPS state for a list of bond dimensions using the CTMRG
    algorithm.

    Args:
        read_fn (str): Filename of the data file to read from.
    """
    args, var_param = read_config()
    section, key = var_param
    var_param_values = args[section][key]
    num_processes = len(var_param_values)

    with mp.Pool(num_processes) as pool:
        pool.starmap(converge, [(var_param, value, args.copy(), read_fn) for value in var_param_values])
