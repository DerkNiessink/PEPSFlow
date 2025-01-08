import rich_click as click

from fabric import Connection
import configparser
import invoke
import os


@click.group()
def server():
    """
    Commands to interact with the remote server.
    """
    pass


@server.command()
def htop():
    """
    Inspection of the server using the htop command.
    """
    conf = configparser.ConfigParser()
    conf.read("pepsflow.cfg")
    address = conf.get("parameters.cli", "server_address").strip("'")
    try:
        with Connection(address) as c:
            c.run("htop", pty=True)
    except invoke.exceptions.UnexpectedExit:
        os.system("cls" if os.name == "nt" else "clear")


@server.command()
@click.argument("PID", type=str)
def kill(pid: str):
    """
    Kill a process on the server.

    PID is the process ID of the process to kill.
    """
    conf = configparser.ConfigParser()
    conf.read("pepsflow.cfg")
    address = conf.get("parameters.cli", "server_address").strip("'")
    with Connection(address) as c:
        c.run(f"kill -SIGINT {pid}")
    print(f"Process {pid} killed.")
