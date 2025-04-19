import configparser
import multiprocessing as mp
import os
import ast
import signal

from pepsflow.iPEPS.iPEPS import make_ipeps
from pepsflow.iPEPS.io import IO
from pepsflow.iPEPS.tools import Tools

path = lambda folder, file: os.path.join("data", folder, file)


def read_config() -> tuple[dict, tuple[str, str]]:
    """
    Read the parameters from the configuration file.

    Returns:
        dict: Dictionary containing the parameters.
        tuple: Tuple containing the section and key of the varying parameter.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = lambda option: option  # Preserve the case of the keys
    parser.read("pepsflow.cfg")

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


def minimize(var_param: tuple[str, str], value: float, args: dict):
    """
    Optimize the iPEPS model for a single value the variational parameter.

    Args:
        var_param (tuple): Section and key of the variational parameter.
        value (float): Value of the variational parameter.
        args (dict): folder, ipeps, and optimization parameters.
    """
    print(f"PID of the task: {os.getpid()}")

    # Set the value of the variational parameter
    section, key = var_param
    args[section][key] = value
    fn = f"{key}_{value}"

    folders = args["parameters.folders"]
    ipeps_params = args["parameters.ipeps"]
    opt_params = args["parameters.optimization"]

    # Read the iPEPS model from a file if specified and set to the device
    if folders["read"]:
        ipeps = IO.load(path(folders["read"], fn))
        ipeps = make_ipeps(args=ipeps_params, initial_ipeps=ipeps)
    else:
        ipeps = make_ipeps(ipeps_params)

    # Save the data if the process is interrupted
    save = lambda sig, frame: (IO.save(ipeps, path(folders["write"], fn)), exit(0))
    signal.signal(signal.SIGINT, save)

    Tools.minimize(ipeps, opt_params)
    IO.save(ipeps, path(folders["write"], fn))


def evaluate(var_param, value: float, args: dict, read_fn: str):
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

    ipeps = IO.load(path(folders["read"], read_fn))

    # Save the data if the process is interrupted
    save = lambda sig, frame: (IO.save(ipeps, path(folders["write"], write_fn)), exit(0))
    signal.signal(signal.SIGINT, save)

    Tools.evaluate(ipeps, ipeps_params)
    IO.save(ipeps, path(folders["write"], write_fn))


def minimize_parallel():
    """
    Optimize the iPEPS model for a list of values of the variational parameter.
    """
    args, var_param = read_config()
    section, key = var_param
    var_param_values = args[section][key]
    num_processes = len(var_param_values)

    with mp.Pool(num_processes) as pool:
        pool.starmap(minimize, [(var_param, value, args.copy()) for value in var_param_values])


def evaluate_parallel(read_fn: str):
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
        pool.starmap(evaluate, [(var_param, value, args.copy(), read_fn) for value in var_param_values])
