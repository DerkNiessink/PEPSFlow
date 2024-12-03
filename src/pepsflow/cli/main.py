import rich_click as click

from pepsflow.cli.data import data
from pepsflow.cli.params import params


@click.group()
def cli():
    """
    pepsflow is a tool to optimize Projected Entangled Pair States (PEPS) using the Corner Transfer Matrix
    Renormalization Group (CTMRG) algorithm and automatic differentiation.
    """
    pass


cli.add_command(data)
cli.add_command(params)


if __name__ == "__main__":
    cli()
