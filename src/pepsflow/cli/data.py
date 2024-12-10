import rich_click as click
import matplotlib.pyplot as plt
import os
from rich.tree import Tree
from rich import print
import pathlib
import shutil
from rich.console import Console
from rich.table import Table
from rich import box
import scienceplots

plt.style.use("science")

from pepsflow.iPEPS.reader import iPEPSReader
from pepsflow.cli.utils import walk_directory

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
@click.argument("folders", type=click.Path(), nargs=-1)
@click.option("-e", "--energy", is_flag=True, default=False, help="Plot the energy as a function of lambda")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Plot the magnetization as a function of lambda")
@click.option("-xi", "--correlation_length", is_flag=True, default=False, help="Plot the correlation length as a function of lambda")
@click.option("-g", "--gradient", type=str, default=None, help="Plot the gradient as a function of epoch. Has to be a .pth file.")
@click.option("-n", "--gradient_norm", type=str, default=None, help="Plot the gradient norm as a function of epoch. Has to be a .pth file.")
@click.option("-c", "--energy_convergence", type=click.Path(), default=None, help="Plot the energy convergence as a function of epoch. Has to be a .json file.")
@click.option("-chi", "--energy_chi", is_flag=True, default=False, help="Plot the converged energy as a function of 1/chi of all .json files in the folder.")
@click.pass_context
def plot(ctx, folders, **kwargs):
    """
    Plot the observables of iPEPS models.

    FOLDER is the folder containing the iPEPS models.
    """
    if sum(bool(opt) for opt in kwargs.values()) > 1:
        ctx.fail("Only one option can be selected at a time.")
    
    all_readers: list[list[iPEPSReader]] = []
    for folder in folders:
        readers = [iPEPSReader(os.path.join("data", folder, x)) for x in os.listdir(os.path.join("data", folder))]
        all_readers.append(readers) 

    plt.figure(figsize=(6, 4))

    if kwargs["magnetization"]:
        plt.ylabel(r"$m_z$", fontsize=12)
        plt.xlabel(r"$\lambda$", fontsize=12)
        symbols = ["^-", "o-"]
        for i, readers in enumerate(all_readers):
            lams, mags = zip(*[(reader.lam(), reader.magnetization()) for reader in readers])
            plt.plot(lams, mags, symbols[i], markersize=5, linewidth=0.5, label=folders[i])
        plt.xlim(2.95, 3.15)
        plt.ylim(-0.0025, 0.45)
        plt.xticks([2.95, 3.0, 3.05, 3.1, 3.15])
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.025))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        plt.tick_params(axis='x', which='minor', length=4)
        plt.grid(which='both', linestyle='--', linewidth=0.5)

    if kwargs["energy"]:
        plt.ylabel(r"$E$")
        plt.xlabel(r"$\lambda$")
        for i, readers in enumerate(all_readers):
            lams, energies = zip(*[(reader.lam(), reader.energy()) for reader in readers])
            plt.plot(lams, energies, "v-", markersize=5, linewidth=0.5, label=rf"${folders[i]}$")

    if kwargs["correlation_length"]:
        plt.ylabel(r"$\xi$")
        plt.xlabel(r"$\lambda$")
        for i, readers in enumerate(all_readers):
            lams, xis = zip(*[(reader.lam(), reader.correlation()) for reader in readers])
            plt.plot(lams, xis, "v-", color="C0", markersize=5, linewidth=0.5, label=f"${folders[i]}$")
        plt.xlim(2.7, 3.3)
        plt.grid(linestyle='--', linewidth=0.5)

    if kwargs["energy_chi"]:
        plt.ylabel(r"$E$")
        plt.xlabel(r"$1/\chi$")
        for i, readers in enumerate(all_readers):
            data = [(1/reader.iPEPS.args["chi"], reader.energy()) for reader in readers if "chi" in reader.file]
            data.sort(reverse=True)
            inv_chis, energies = zip(*data)
            plt.plot(inv_chis, energies, "v-", markersize=4, linewidth=0.5, label=folders[i])

    if kwargs["gradient"]:
        plt.ylabel(r"$E$")
        plt.xlabel(r"Epoch")
        for file in kwargs["gradient"].split(","):
            reader = iPEPSReader(os.path.join("data", folder, file))
            losses = reader.losses()
            plt.plot(range(len(losses)), losses, "v-", markersize=4, linewidth=0.5, label=rf'${file}$')
        plt.ylim(-0.67, -0.64)

    if kwargs["gradient_norm"]:
        plt.ylabel(r"$\| \nabla E \|$")
        plt.xlabel(r"Epoch")
        for file in kwargs["gradient_norm"].split(","):
            reader = iPEPSReader(os.path.join("data", folder, file))
            norms = reader.gradient_norms()
            plt.plot(range(len(norms)), norms, "v-", markersize=4, linewidth=0.5, label=reader.file)

        
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/Ising_xi_D2.png")
    plt.show()


@data.command()
@click.argument("old", type=str)
@click.argument("new", type=str)
def rename(old: str, new: str):
    """
    Rename a folder name to a new name.

    OLD is the current folder name. NEW is the new folder name.
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

    PATH is the file or folder to remove.
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
@click.option("-p", "--params", is_flag=True, default=False, help="Print the parameters of the iPEPS model.")
def info(folder: click.Path, file: click.Path, state: bool, lam: bool, energy: bool, magnetization: bool, correlation: bool, losses: bool, params: bool):
    """
    Print the information of the iPEPS models in the specified FOLDER.

    FOLDER contains the iPEPS models.
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
    if params: table.add_column("iPEPS Parameters", justify="left")

    filenames = [file] if file else os.listdir(os.path.join("data", folder))
    
    for i, f in enumerate(filenames):
        reader = iPEPSReader(os.path.join("data", folder, f))
        row = [f]
        if energy or print_all: row.append(f"{reader.energy()}")
        if magnetization or print_all: row.append(f"{reader.magnetization()}")
        if correlation or print_all: row.append(f"{reader.correlation()}")
        if losses: row.append(f"{reader.losses()}")
        if state: row.append(f"{reader.iPEPS_state()}")
        if params: row.append(f"{reader.iPEPS.args}")
        style = "grey50" if i % 2 != 0 else "grey78"
        table.add_row(*row, style=style)

    console.print(table)
    console.print("\n")


@data.command()
@click.argument("folder", type=click.Path())
@click.option("-f", "--file", default=None, type=click.Path(), help="File containing data, if not specified, all files in the folder are set to the lowest energy.")
def lowest(folder: click.Path, file: click.Path):
    """
    Set the last epoch of the iPEPS model to the lowest energy.

    FOLDER contains the iPEPS models.
    """
    filenames = [file] if file else os.listdir(os.path.join("data", folder))
    for f in filenames:
        reader = iPEPSReader(os.path.join("data", folder, f))
        reader.set_to_lowest_energy()
