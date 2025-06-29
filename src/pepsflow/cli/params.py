import rich_click as click
import configparser
from rich.console import Console
from rich.table import Table
from rich import box
from fabric import Connection
import subprocess

from pepsflow.cli.utils import read_cli_config
from pepsflow.pepsflow import Pepsflow

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
        config.read("pepsflow.cfg")
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
@click.option("--server", "-s", is_flag=True, help="Whether to run the optimization on thea server.")
def optimize(server: bool = False):
    """
    Optimize the iPEPS tensor network with the specified parameters in the configuration file.
    """
    if server:
        args = read_cli_config()
        data, write = args['data'], args['write']
        subprocess.run(["scp", "pepsflow.cfg", f"{args['server_address']}:PEPSFlow/pepsflow.cfg"])
        with Connection(args['server_address']) as c:
            print("Running the optimization...")
            c.run(f"mkdir -p PEPSFlow/{data}/{write}")
            c.run(
                f"cd PEPSFlow && source .venv/bin/activate && "
                f"mkdir -p {data}/{write} && "
                f"bash -c 'nohup pepsflow params optimize > {data}/{write}/{write}_$$.out 2>&1 & disown'"
            )
    else: 
        Pepsflow().optimize_parallel()


@params.command()
@click.argument("filename", type=str)
def evaluate(filename: str):
    """
    Evaluate the iPEPS tensor network with the specified parameters in the configuration file.

    FILENAME is the data file to read from.
    """
    Pepsflow().evaluate(filename)


@params.command()
@click.argument("filename", type=str)
def gauge(filename: str):
    """
    Apply gauge transformations to the iPEPS tensor network with the specified parameters in the 
    configuration file.

    FILENAME is the data file to read from.
    """
    Pepsflow().gauge(filename)
