import click

from pepsflow.cli.data import data
from pepsflow.cli.params import params


@click.group()
def cli():
    pass


cli.add_command(data)
cli.add_command(params)


if __name__ == "__main__":
    cli()
