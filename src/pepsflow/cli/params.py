import rich_click as click
import configparser
from rich.console import Console
from rich.table import Table
from rich import box
import ast

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
        config.optionxform = lambda option: option # Preserve case
        config.read("src/pepsflow/pepsflow.cfg")
        console = Console()
        table = Table(title="PEPSFLOW parameters", title_justify="center", box=box.SIMPLE)
        table.add_column("Parameter", no_wrap=True, justify="left")
        table.add_column("Value")
        for section in config.sections():
            table.add_section()
            if section != "parameters":
                table.add_row(f"[bold]{section}", "", style="bold underline")
            for i, (key, value) in enumerate(config.items(section)):
                style = "grey50" if i % 2 != 0 else "grey78"  
                table.add_row(key, value, style = style)
        console.print(table)


@params.command()
@click.option("-N","--Niter","Niter", type=int, help="Number of iterations in the forward step.")
@click.option("--D", "D", type=str, help="One value or comma separated list of values of the bulk dimension d of the iPEPS tensor.")
@click.option("--model", type=str, help="Model to optimize. Options are 'Ising', 'Heisenberg', and 'J1J2'.")
@click.option("--chi", type=str, help="One value of comma separated list of values of the bond dimension chi of the CTM algorithm.")
@click.option("-r","--read", type=str, help="Folder containing iPEPS models and other datafiles to read.")
@click.option("--gpu/--no-gpu", default=None, type=bool, help="Run the model on the GPU if available.")
@click.option("--lam", type=str, help="One value of comma separated list of values of the parameter lambda in the tranverse-field Ising model.")
@click.option("-lr","--learning_rate", type=str, help="One value or comma separated list of values of the learning rate for the optimizer.")
@click.option("--epochs", type=int, help="Maximum number of epochs to train the model.")
@click.option("-per", "--perturbation", type=float, help="Amount of perturbation to apply to the initial state.")
@click.option("-w", "--write", type=str, help="Folder to save the iPEPS model in.")
@click.option("--threads", type=int, help="Number of threads to use for the optimization. Each thread runs on a separate CPU core.")
@click.option("-ws", "--warmup_steps", type=int, help="Number of warmup steps to perform in the CTM algorithm before starting the optimization. This is only applicable if no previous data file is given.")
@click.option("-ls/-no-ls", "--line_search/--no-line_search", default=None, type=bool, help="Use Wolfe line search in the LBFGS optimizer.")
@click.option("-s", "--start_epoch", type=int, help="Epoch to start the optimization from. This is only applicable if a previous data file is given. If -1, the optimization starts from the last epoch.")
@click.option("--split/--no-split", default = None, type=bool, help="Keep the tensor in the CTM algorithm split or not.")
@click.option("-o", "--optimizer", type=str, help="Optimizer to use. Options are 'adam' and 'lbfgs'.")
@click.option("--seed", type=float, help="Seed for the random generation of tensors.")
def set(Niter: int, D: int, **args):
    """
    Set specific parameters for the optimization of the iPEPS tensor network.
    """
    args['Niter'], args['D'] = Niter, D # For case preservation.
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    file = "src/pepsflow/pepsflow.cfg"
    config.read(file)

    for section in config.sections():
        for param, value in args.items():
            if param in config[section] and value is not None:
                
                # Convert to list if comma separated values
                if type(value) == str and ',' in value:
                    value = [ast.literal_eval(x) for x in value.split(",") if x]

                config[section][param] = str(value)

    with open(file, 'w') as configfile: config.write(configfile)


@params.command()
def optimize():
    """
    Optimize the iPEPS tensor network with the specified parameters in the configuration file.
    """
    pepsflow.optimize_parallel()


@params.command()
@click.argument("filename", type=str)
def converge(filename: str):
    """
    Converge the iPEPS tensor network with the specified parameters in the configuration file.

    FILENAME is the data file to read from.
    """
    pepsflow.converge_parallel(filename)
