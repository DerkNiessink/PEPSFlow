import click
import matplotlib.pyplot as plt
import os
from rich.tree import Tree
from rich import print
import pathlib
import shutil
from rich.console import Console
from rich.table import Table
from rich import box
import json

from pepsflow.iPEPS.reader import iPEPSReader
from pepsflow.cli.utils import get_observables, walk_directory
from matplotlib.ticker import LogLocator

# fmt: off


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


@data.command(context_settings={"ignore_unknown_options": True})
@click.argument("folders", nargs=-1, type=click.Path())
@click.option("-e", "--energy", is_flag=True, default=False, help="Plot the energy as a function of lambda")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Plot the magnetization as a function of lambda")
@click.option("-xi", "--correlation_length", is_flag=True, default=False, help="Plot the correlation length as a function of lambda")
@click.option("-g", "--gradient", type=click.Path(), default=None, help="Plot the gradient as a function of epoch. Has to be a .pth file.")
@click.option("-n", "--gradient_norm", type=click.Path(), default=None, help="Plot the gradient norm as a function of epoch. Has to be a .pth file.")
@click.option("-c", "--energy_convergence", type=click.Path(), default=None, help="Plot the energy convergence as a function of epoch. Has to be a .json file.")
@click.option("-chi", "--energy_chi", type=click.Path(), default=None, help="Plot the converged energy as a function of 1/chi. Has to be a .json file.")
def plot(folders: click.Path, correlation_length: bool, energy: bool, magnetization: bool, gradient: click.Path, gradient_norm: click.Path, energy_convergence: click.Path, energy_chi: click.Path):
    """
    Plot the observables of the iPEPS models.
    """    
    if magnetization:
        mag_figure, mag_ax = plt.subplots(figsize=(6, 4))
        mag_ax.set_ylabel(r"$\langle M_z \rangle$")
        mag_ax.set_xlabel(r"$\lambda$")

    if energy:
        en_figure, en_ax = plt.subplots(figsize=(6, 4))
        en_ax.set_ylabel(r"$E$")
        en_ax.set_xlabel(r"$\lambda$")

    if correlation_length:
        xi_figure, xi_ax = plt.subplots(figsize=(6, 4))
        xi_ax.set_ylabel(r"$\xi$")
        xi_ax.set_xlabel(r"$\lambda$")

    for folder in folders:
        data = get_observables(folder, magnetization, energy, correlation_length, gradient, gradient_norm, energy_convergence, energy_chi)

        if magnetization:
            mag_ax.plot(data["lam"], data["M"], "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if energy:
            en_ax.plot(data["lam"], data["E"], "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if correlation_length:
            xi_ax.plot(data["lam"], data["xi"], "v-", markersize=4, linewidth=0.5, label=rf"${folder}$")

        if gradient:
            grad_figure, grad_ax = plt.subplots(figsize=(6, 4))
            grad_ax.plot(range(len(data["losses"])), data["losses"], "v-", markersize=4, linewidth=0.5, label=folder)
            grad_ax.set_ylabel("$E$")
            grad_ax.set_xlabel("Epoch")
            grad_ax.legend()
            grad_ax.grid(which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.show()

        if gradient_norm:
            grad_norm_figure, grad_norm_ax = plt.subplots(figsize=(6, 4))
            grad_norm_ax.plot(range(len(data["norms"])), data["norms"], "v-", markersize=4, linewidth=0.5, label=folder)
            grad_norm_ax.set_ylabel("Gradient Norm")
            grad_norm_ax.set_xlabel("Epoch")
            grad_norm_ax.legend()
            plt.show()

        if energy_convergence:
            energy_convergence_figure, energy_convergence_ax = plt.subplots(figsize=(6, 4))
            for vals in data["energy_convergence"]:
                energy_convergence_ax.plot(range(len(vals["energies"])), vals["energies"], "v-", markersize=4, linewidth=0.5, label=rf"$\chi={vals['chi']}$")
            energy_convergence_ax.set_ylabel("$E$")
            energy_convergence_ax.set_xlabel("CTMRG Steps")
            energy_convergence_ax.legend()
            energy_convergence_ax.grid(which='both', linestyle='--', linewidth=0.5)	
            energy_convergence_ax.set_xlim(0, 30)
            energy_convergence_ax.set_ylim(-0.6659,-0.6645)
            plt.tight_layout()
            plt.show()

        if energy_chi:
            energy_chi_figure, energy_chi_ax = plt.subplots(figsize=(6, 4))
            inverse_chis, energies = [], []
            for vals in data["energy_convergence"]:
                inverse_chis.append(1/vals["chi"])
                energies.append(vals["energies"][-1])
            
            energy_chi_ax.set_xlim(0, 1.05*max(inverse_chis))
            energy_chi_ax.grid(which='both', linestyle='--', linewidth=0.5)
            energy_chi_ax.plot(inverse_chis, energies, "v-", markersize=4, linewidth=0.5, label=f"${folder}$")
            energy_chi_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))
            energy_chi_ax.set_ylabel("$E$")
            energy_chi_ax.set_xlabel(r"$1/\chi$")
            energy_chi_ax.legend()
            plt.tight_layout()
            plt.show()
    

    if magnetization:
        mag_ax.legend()
        plt.show()

    if energy:
        en_ax.legend()
        plt.show()

    if correlation_length:
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
@click.argument("path", type=click.Path())
def remove(path: click.Path):
    """
    Remove a file or folder from the data directory.
    """
    full_path = os.path.join("data", path) 
    if os.path.isdir(full_path): 
        message = f"\nAre you sure you want to remove the folder [red]{path}[/] and all its contents?"
        remove_func = shutil.rmtree
    else: 
        message = f"\nAre you sure you want to remove [red]{path}?" 
        remove_func = os.remove

    print(message)
    if click.confirm(""):
        remove_func(full_path)
        print(f"\nRemoved [red]{path}")
    else:
        print("\nCancelled") 


@data.command()
@click.argument("folder", type=click.Path())
@click.option("-f", "--file", default=None, type=click.Path(), help="File containing data, if not specified, all files in the folder are printed.")
@click.option("-s", "--state", is_flag=True, default=False, help="Print the iPEPS state.")
@click.option("-l", "--lam", is_flag=True, default=False, help="Print the lambda value.")
@click.option("-e", "--energy", is_flag=True, default=False, help="Print the energy.")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Print the magnetization.")
@click.option("-xi", "--correlation", is_flag=True, default=False, help="Print the correlation.")
@click.option("-o", "--losses", is_flag=True, default=False, help="Print the losses.")
def info(folder: click.Path, file: click.Path, state: bool, lam: bool, energy: bool, magnetization: bool, correlation: bool, losses: bool):
    """
    Print the information of the iPEPS models in the specified folder.
    """
    console = Console()
    console.print("\n")
    table = Table(title=f"iPEPS Information for Folder: {folder}", box=box.MINIMAL_DOUBLE_HEAD)
    
    print_all = not any([lam, energy, magnetization, correlation, losses, state])

    table.add_column("Filename", justify="right", no_wrap=True, style="blue bold")
    if energy or print_all: table.add_column("Energy", justify="right")
    if magnetization or print_all: table.add_column("Magnetization", justify="right")
    if correlation or print_all: table.add_column("Correlation", justify="right")
    if losses: table.add_column("Losses", justify="left")
    if state: table.add_column("State", justify="left")

    filenames = [file] if file else os.listdir(os.path.join("data", folder))
    
    for i, f in enumerate(filenames):
        reader = iPEPSReader(os.path.join("data", folder, f))
        row = [f]
        if energy or print_all: row.append(f"{reader.energy()}")
        if magnetization or print_all: row.append(f"{reader.magnetization()}")
        if correlation or print_all: row.append(f"{reader.correlation()}")
        if losses: row.append(f"{reader.losses()}")
        if state: row.append(f"{reader.iPEPS_state()}")
        style = "grey50" if i % 2 != 0 else "grey78"
        table.add_row(*row, style=style)

    console.print(table)
    console.print("\n")
