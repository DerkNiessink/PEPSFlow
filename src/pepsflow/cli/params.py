import click
import configparser
from rich.console import Console
from rich.table import Table
from rich import box

import pepsflow.pepsflow as pepsflow

# fmt: off


@click.group(invoke_without_command=True)
@click.pass_context
def params(ctx):
    """
    Show the current parameters for the optimization of the iPEPS tensor network.
    """
    if ctx.invoked_subcommand is None:
        config = configparser.ConfigParser()
        config.optionxform = str # Preserve the case of the keys
        config.read("src/pepsflow/pepsflow.cfg")
        console = Console()
        table = Table(title="Optimization parameters", title_justify="center", box=box.SIMPLE_HEAVY)
        table.add_column("Parameter", no_wrap=True, justify="left")
        table.add_column("Value", style="green")
        for i, (key, value) in enumerate(config['PARAMETERS'].items()):
            style = "grey50" if i % 2 != 0 else "grey78"
                
            table.add_row(key, value, style = style)
        console.print(table)


@params.command()
@click.option("--model", type=str, help="Model to optimize. Options are 'Ising' and 'Heisenberg'")
@click.option("--chi", type=str, help="Comma separated list of values of the bond dimension chi of the CTM algorithm")
@click.option("--D", type=str, help="Comma separated list of values of the bulk dimension d of the iPEPS tensor")
@click.option("-df","--data_folder", type=str, help="Folder containing iPEPS models")
@click.option("--gpu/--no-gpu", default=None, type=bool, help="Run the model on the GPU if available")
@click.option("--lam", type=str, help="Comma separated list of values of the parameter lambda in the tranverse-field Ising model")
@click.option("--runs", type=int, help="Number of runs to train the model. Applies to random initialization. The program will choose the best model based on the lowest energy.")
@click.option("-lr","--learning_rate", type=float, help="Learning rate for the optimizer")
@click.option("--epochs", type=int, help="Maximum number of epochs to train the model")
@click.option("-per", "--perturbation", type=float, help="Amount of perturbation to apply to the initial state")
@click.option("-sf", "--save_folder", type=str, help="Folder to save the iPEPS model in.")
@click.option("--threads", type=int, help="Number of threads to use for the optimization. Each thread runs on a separate CPU core.")
@click.option("-ws", "--warmup_steps", type=int, help="Number of warmup steps to perform in the CTM algorithm before starting the optimization. This is only applicable if no previous data file is given.")
@click.option("-ls/-no-ls", "--line_search/--no-line_search", default=None, type=bool, help="Use Wolfe line search in the LBFGS optimizer.")
@click.option("-s", "--start_epoch", type=int, help="Epoch to start the optimization from. This is only applicable if a previous data file is given. If -1, the optimization starts from the last epoch.")
@click.option("--split/--no-split", default = None, type=bool, help="Keep the tensor in the CTM algorithm split or not.")
@click.option("-o", "--optimizer", type=str, help="Optimizer to use. Options are 'adam' and 'lbfgs'")
def set(**args):
    """
    Set specific parameters for the optimization of the iPEPS tensor network.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    file = "src/pepsflow/pepsflow.cfg"
    config.read(file)    

    # Regular expression pattern to match parameters
    for param, value in args.items():
        if value is not None and value != ():
            # Case sensitive parameters
            param = 'D' if param == 'd' else param
            # Varying parameters
            if param == 'lam' and ',' in value:
                value = [float(x) for x in value.split(',') if x]
            if (param == 'chi' or param == 'D') and ',' in value:
                value = [int(x) for x in value.split(',') if x]
            config['PARAMETERS'][param] = str(value)

    config.optionxform = str
    with open(file, 'w') as configfile: config.write(configfile)


@params.command()
def optimize():
    """
    Optimize the iPEPS tensor network with the specified parameters in the configuration file.
    """
    pepsflow.optimize()


@params.command()
@click.argument("filename", type=str)
def converge(filename: str):
    """
    Converge the iPEPS tensor network with the specified parameters in the configuration file.

    Args:
        filename (str): Filename of the data to read.
    """
    pepsflow.converge(filename)
