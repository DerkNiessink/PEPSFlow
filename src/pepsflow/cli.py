import click
import matplotlib.pyplot as plt
import os
import configparser
from rich.console import Console
from rich.table import Table
from rich import box
from rich.tree import Tree
from rich import print
import pathlib
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape

from pepsflow.train.iPEPS_reader import iPEPSReader
import pepsflow.optimize

# fmt: off

@click.group()
def cli():
    pass


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--folder", "-f", default = None, type=str, help="Show the data files in the folder.")
@click.option("--concise", "-c", is_flag=True, default = False, help="Only show the folder names.")
def data(ctx, folder: str, concise: bool):
    """
    List the data files in the data folder.
    """
    if ctx.invoked_subcommand is None:
        directory = pathlib.Path("data", folder) if folder else pathlib.Path("data")
        tree = Tree(
        f":open_file_folder: {directory}",
        )
        walk_directory(pathlib.Path(directory), tree, concise)
        print(tree)



def walk_directory(directory: pathlib.Path, tree: Tree, concise) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold]:open_file_folder: {escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch, concise)
        elif not concise:
            text_filename = Text(path.name)
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            icon = "🐍 " if path.suffix == ".py" else "📄 "
            tree.add(Text(icon) + text_filename)



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


cli.add_command(data)
cli.add_command(params)

@params.command()
def optimize():
    """
    Optimize the iPEPS tensor network with the specified parameters in the configuration file.
    """
    pepsflow.optimize.optimize()


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


@data.command(context_settings={"ignore_unknown_options": True})
@click.argument("folders", nargs=-1, type=click.Path())
@click.option("-e", "--energy", is_flag=True, default=False, help="Plot the energy as a function of lambda")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Plot the magnetization as a function of lambda")
@click.option("-xi", "--correlation_length", is_flag=True, default=False, help="Plot the correlation length as a function of lambda")
@click.option("-g", "--gradient", type=click.Path(), default=None, help="Plot the gradient as a function of epoch")
def plot(folders: click.Path, correlation_length: bool, energy: bool, magnetization: bool, gradient: click.Path):
    """
    Plot the observables of the iPEPS models.
    """
    plot_all = not any([magnetization, energy, correlation_length, gradient])
    
    if magnetization or plot_all:
        mag_figure, mag_ax = plt.subplots(figsize=(6, 4))
        mag_ax.set_ylabel(r"$\langle M_z \rangle$")
        mag_ax.set_xlabel(r"$\lambda$")

    if energy or plot_all:
        en_figure, en_ax = plt.subplots(figsize=(6, 4))
        en_ax.set_ylabel(r"$E$")
        en_ax.set_xlabel(r"$\lambda$")

    if correlation_length or plot_all:
        xi_figure, xi_ax = plt.subplots(figsize=(6, 4))
        xi_ax.set_ylabel(r"$\xi$")
        xi_ax.set_xlabel(r"$\lambda$")

    for folder in folders:
        reader = iPEPSReader(os.path.join("data", folder))
        lambda_values = reader.get_lambdas()

        if magnetization or plot_all:
            mag_ax.plot(lambda_values, reader.get_magnetizations(), "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if energy or plot_all:
            en_ax.plot(lambda_values, reader.get_energies(), "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if correlation_length or plot_all:
            xi_ax.plot(lambda_values, reader.get_correlations(), "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if gradient:
            grad_figure, grad_ax = plt.subplots(figsize=(6, 4))
            losses = reader.get_losses(gradient)
            grad_ax.plot(range(len(losses)), losses, "v-", markersize=4, linewidth=0.5, label=folder)
            grad_ax.set_ylabel("$E$")
            grad_ax.set_xlabel("Epoch")
            grad_ax.legend()
            plt.show()

    if magnetization or plot_all:
        mag_ax.legend()
        plt.show()

    if energy or plot_all:
        en_ax.legend()
        plt.show()

    if correlation_length or plot_all:
        xi_ax.legend()
        plt.show()


@data.command()
@click.argument("old", type=str)
@click.argument("new", type=str)
def rename(old: str, new: str):
    """
    Rename a folder name to a new name.
    """
    for dirpath, dirnames, filenames in os.walk("data"):
        if os.path.basename(dirpath) == old:
            os.rename(dirpath, os.path.join("data", new))
            break


@data.command()
@click.argument("folder", type=click.Path())
@click.option("-f", "--file", default=None, type=click.Path(), help="File containing data, if not specified, all files in the folder are printed.")
def state(folder: click.Path, file: click.Path):
    """
    Print the tensors of the iPEPS model in the specified folder.
    """
    reader = iPEPSReader(os.path.join("data", folder))
    filenames = [file] if file else reader.filenames
    for f in filenames:
        print(reader.get_iPEPS_state(f))
