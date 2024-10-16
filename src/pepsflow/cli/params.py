import click
import configparser
from rich.console import Console
from rich.table import Table
from rich import box

import pepsflow.optimize

# fmt: off


@click.group(invoke_without_command=True)
@click.pass_context
def params(ctx):
    """
    Show the current parameters for the optimization of the iPEPS tensor network.
    """
    if ctx.invoked_subcommand is None:
        config = configparser.ConfigParser()
        config.read("src/pepsflow/optimize.cfg")
        console = Console()
        table = Table(title="Optimization parameters", title_justify="center", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Parameter", no_wrap=True)
        table.add_column("Value", style="green")
        for key, value in config['PARAMETERS'].items():
            table.add_row(key, value)
        console.print(table)


@params.command()
@click.option("--model", type=str, help="Model to optimize. Options are 'Ising' and 'Heisenberg'")
@click.option("--chi", type=int, help="Environment bond dimension in the CTMRG algorithm")
@click.option("--d", type=int, help="Bulk bond dimension of the iPEPS")
@click.option("-df","--data_folder", type=str, help="Folder containing iPEPS models")
@click.option("--gpu/--no-gpu", type=bool, help="Run the model on the GPU if available")
@click.option("--lam", type=str, help="Comma separated list of values of the parameter lambda in the tranverse-field Ising model")
@click.option("--max_iter", type=int, help="Maximum number of iterations for the optimizer")
@click.option("--runs", type=int, help="Number of runs to train the model. Applies to random initialization. The program will choose the best model based on the lowest energy.")
@click.option("-lr","--learning_rate", type=float, help="Learning rate for the optimizer")
@click.option("--epochs", type=int, help="Maximum number of epochs to train the model")
@click.option("-per", "--perturbation", type=float, help="Amount of perturbation to apply to the initial state")
@click.option("-sf", "--save_folder", type=str, help="Folder to save the iPEPS model in.")
@click.option("--threads", type=int, help="Number of threads to use for the optimization. Each thread runs on a separate CPU core.")
def set(**args):
    """
    Set specific parameters for the optimization of the iPEPS tensor network.
    """
    config = configparser.ConfigParser()
    file = "src/pepsflow/optimize.cfg"
    config.read(file)    

    # Regular expression pattern to match parameters
    for param, value in args.items():
        if value is not None and value != ():
            if param == 'lam':
                value = [float(x) for x in value.split(',')]
            
            config['PARAMETERS'][param] = str(value)

    config.write(open(file, 'w'))


@params.command()
def optimize():
    """
    Optimize the iPEPS tensor network with the specified parameters in the configuration file.
    """
    pepsflow.optimize.optimize()
