import rich_click as click
from fabric import Connection
import configparser
import invoke
import os

from pepsflow.cli.data import data
from pepsflow.cli.params import params


@click.group()
def cli():
    """
    pepsflow is a tool to optimize Projected Entangled Pair States (PEPS) using the Corner Transfer Matrix
    Renormalization Group (CTMRG) algorithm and automatic differentiation.
    """
    pass


@cli.command()
def server():
    """
    Inspection of the server using the htop command.
    """
    c = configparser.ConfigParser()
    c.read("src/pepsflow/pepsflow.cfg")
    address = c.get("parameters.cli", "server_address").strip("'")
    try:
        with Connection(address) as c:
            c.run("htop", pty=True)
    except invoke.exceptions.UnexpectedExit:
        os.system("cls" if os.name == "nt" else "clear")


cli.add_command(data)
cli.add_command(params)


if __name__ == "__main__":
    cli()
