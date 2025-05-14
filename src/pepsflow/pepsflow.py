import configparser
import multiprocessing as mp
import os
import ast
import signal

from pepsflow.ipeps.ipeps import make_ipeps, iPEPS
from pepsflow.ipeps.io import IO
from pepsflow.ipeps.tools import Tools


class Pepsflow:
    """
    Class to handle the workflow of the iPEPS optimization, evaluation, and gauge transformation.
    It is responsible for reading the configuration file, setting up the iPEPS model, and managing the
    optimization and evaluation processes.

    args:
        config_path (str): Path to the configuration file. Default is "pepsflow.cfg".
    """

    def __init__(self, config_path: str = "pepsflow.cfg"):
        self.config_path = config_path

        self.folders = self._read_config("folders")
        self.optimization_args = self._read_config("tools.optimize.optimizer")
        self.ipeps_args = self._read_config("tools.optimize.ipeps")
        self.evaluate_args = self._read_config("tools.evaluate")
        self.gauge_args = self._read_config("tools.gauge")

    def _read_config(self, section: str) -> dict:
        """
        Read the configuration file and return the parameters as a dictionary.

        Args:
            section (str): The section of the configuration file to read.

        Returns:
            dict: Dictionary containing the parameters from the specified section.
        """
        parser = configparser.ConfigParser()
        parser.optionxform = lambda option: option  # Preserve the case of the keys
        parser.read(self.config_path)
        if not parser.sections():
            raise FileNotFoundError(f"Configuration file {self.config_path} is empty or not found.")

        args = {}

        # Try to evaluate the value as a Python literal (e.g., int, float, list, etc.)
        # If it fails, keep it as a string
        for key, value in parser.items(section):
            try:
                value = ast.literal_eval(value)
            except ValueError:
                pass
            args[key] = value

        return args

    def _path(self, folder: str, filename: str) -> str:
        """
        Construct a full path from a folder and filename.
        """
        return os.path.join(self.folders["data"], folder, filename)

    def _handle_interrupt(self, ipeps: iPEPS, save_path: str) -> None:
        """
        Define a SIGINT handler that saves the current iPEPS state.
        """
        handler = lambda sig, frame: (IO.save(ipeps, save_path), exit(0))
        signal.signal(signal.SIGINT, handler)

    def optimize(self, key: str, value: float) -> None:
        """
        Optimize the iPEPS model for a single value of the variational parameter.

        Args:
            key (str): The variational parameter key (e.g., "chi").
            value (float): Value of the variational parameter.
        """
        print(f"PID: {os.getpid()}")

        # The variational parameter can be in either the iPEPS or optimization arguments
        if key in self.ipeps_args:
            self.ipeps_args[key] = value
        else:
            self.optimization_args[key] = value

        filename = f"{key}_{value}"

        # Load the given iPEPS state and change the parameters to the values in the config file
        initial_ipeps = IO.load(self._path(self.folders["read"], filename)) if self.folders["read"] else None
        ipeps = make_ipeps(self.ipeps_args, initial_ipeps)

        write_path = self._path(self.folders["write"], filename)
        self._handle_interrupt(ipeps, write_path)
        Tools.minimize(ipeps, self.optimization_args)
        IO.save(ipeps, write_path)

    def evaluate(self, read_filename: str) -> None:
        """
        Evaluate the energy of a converged iPEPS state using the CTMRG algorithm.

        Args:
            read_filename (str): Filename of the converged iPEPS state to evaluate.
        """
        read_path = self._path(self.folders["read"], read_filename)
        ipeps = IO.load(read_path)
        Tools.evaluate(ipeps, self.evaluate_args)
        IO.save(ipeps, read_path)

    def gauge(self, read_filename: str) -> None:
        """
        Apply a gauge transformation to a saved iPEPS state.

        Args:
            read_filename (str): Filename of the iPEPS state to transform.
        """
        read_path = self._path(self.folders["read"], read_filename)
        write_path = self._path(self.folders["write"], read_filename)
        ipeps = IO.load(read_path)
        Tools.gauge(ipeps, self.gauge_args)
        IO.save(ipeps, write_path)

    def optimize_parallel(self) -> None:
        """
        Optimize multiple iPEPS models in parallel, sweeping over a list of parameter values.
        """
        # Find the variational parameter (must be a list)
        get_var_param_key = lambda args: next((key for key, value in args.items() if isinstance(value, list)), None)
        var_param_key = get_var_param_key(self.ipeps_args) or get_var_param_key(self.optimization_args)

        if not var_param_key:
            raise KeyError("No variational parameter found.")

        # Determine which dictionary contains the variational parameter
        var_param_dict = self.ipeps_args if var_param_key in self.ipeps_args else self.optimization_args

        values = var_param_dict[var_param_key]
        num_processes = len(values)

        with mp.Pool(num_processes) as pool:
            pool.starmap(Pepsflow._optimize_wrapper, [(var_param_key, value, self.config_path) for value in values])

    @staticmethod
    def _optimize_wrapper(key, value, config_path):
        """Wrapper for the optimize function to be used with multiprocessing."""
        Pepsflow(config_path).optimize(key, value)
