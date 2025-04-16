import rich_click as click
import matplotlib.pyplot as plt
import matplotlib
import os
from rich.tree import Tree
from rich import print
import pathlib
import shutil
from rich.table import Table
from rich import box
from fabric import Connection
import subprocess
import numpy as np

from pepsflow.iPEPS.reader import Reader
from pepsflow.cli.utils import walk_directory, read_cli_config

# fmt: off


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--folder", "-f", default = None, type=str, help="Show the data files in the folder.")
@click.option("--concise", "-c", is_flag=True, default = False, help="Only show the folder names.")
@click.option("--server", "-s", is_flag=True, default = False, help="Show the data files in the server.")
def data(ctx, folder: str, concise: bool, server: bool):
    """
    List the data files in the data folder.
    """
    if ctx.invoked_subcommand is None:
        args = read_cli_config()
        if server:
            with Connection(args["server_address"]) as c:
                c.run(f"cd PEPSFlow && git restore . && git pull", hide=True)
                c.run(f"cd PEPSFlow && source .venv/bin/activate && pepsflow data")
        else:
            directory = pathlib.Path(args["data"], folder) if folder else args["data"]
            tree = Tree(f"{directory}")
            tree.TREE_GUIDES = [("    ", "|   ", "+-- ", "`-- ")]
            walk_directory(pathlib.Path(directory), tree, concise)
            print(tree)

@data.command()
@click.argument("folders", type=click.Path(), nargs=-1)
@click.option("-s", "--server", is_flag=True, default=False, help="Wether to copy the data files from the server.")
def copy(folders: str, server: bool):
    """
    Copy the data files from the server to the local machine, or vice versa if the --server flag is used.

    FOLDERS are the folders in the data directory to copy 
    """
    args = read_cli_config()
    for folder in folders:
        if server:
            subprocess.run(["scp","-r", f"{args['data']}/{folder}", f"{args['server_address']}:PEPSFlow/{args['data']}"])
        else:
            subprocess.run(["scp","-r", f"{args['server_address']}:PEPSFlow/{args['data']}/{folder}", args['data']])
            

@data.command()
@click.argument("folder", type=click.Path())
@click.option("--server", "-s", is_flag=True, default=False, help="Wether to print the log file from the server.")
def log(folder: str, server = False):
    """
    Show the log file of the data folder.

    FOLDER is the folder containing the log file.
    """
    args = read_cli_config()
    if server:
        with Connection(args["server_address"]) as c:
            c.run(f"cd PEPSFlow && git restore . && git pull", hide=True)
            c.run(f"cd PEPSFlow && source .venv/bin/activate && pepsflow data log {folder}")
    else:
        with open(f"{args['data']}/{folder}/{folder}.out") as f:
            print(f.read())

@data.command(context_settings={"ignore_unknown_options": True})
@click.argument("folders", type=click.Path(), nargs=-1)
@click.option("-e", "--energy", is_flag=True, default=False, help="Plot the energy as a function of lambda")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Plot the magnetization as a function of lambda")
@click.option("-xi", "--correlation_length", is_flag=True, default=False, help="Plot the correlation length as a function of lambda")
@click.option("-g", "--gradient", type=str, default=None, help="Plot the gradient as a function of epoch. Has to be a .pth file.")
@click.option("-n", "--gradient_norm", type=str, default=None, help="Plot the gradient norm as a function of epoch. Has to be a .pth file.")
@click.option("-c", "--energy_convergence", type=click.Path(), default=None, help="Plot the energy convergence as a function of epoch. Has to be a .pth file.")
@click.option("-chi", "--energy_chi", is_flag=True, default=False, help="Plot the converged energy as a function of 1/chi of all .pth files in the folder.")
@click.option("-ctm", "--ctmsteps", type=click.Path(), default=None, help="Plot the number of CTM warmup steps each epoch. Has to be a .pth file.")
@click.option("-ep", "--epochs", is_flag=True, default=False, help="Plot the number of epochs as a function of the number of gradient steps.")
@click.option("-f", "--final_energies", is_flag=True, default=False, help="Plot the final energy as a function of the number of gradient and warmup steps")
@click.option("-w", "--warmup_steps", is_flag=True, default=False, help="Plot the final energy as a function of the number of warmup steps")
@click.pass_context
def plot(ctx, folders, **kwargs):
    """
    Plot the observables of iPEPS models.

    FOLDERS are the folders containing the iPEPS models.
    """
    args = read_cli_config()
    if args["latex"]:
        import scienceplots
        plt.style.use("science")

    if sum(bool(opt) for opt in kwargs.values()) > 1:
        ctx.fail("Only one option can be selected at a time.")
    
    all_readers: list[list[Reader]] = []
    for folder in folders:
        readers = [Reader(os.path.join("data", folder, x)) for x in os.listdir(os.path.join("data", folder)) if x.endswith(".pth")]
        all_readers.append(readers) 

    plt.figure(figsize=(6, 4))

    if kwargs["magnetization"]:
        plt.ylabel(r"$m_z$", fontsize=12)
        plt.xlabel(r"$\lambda$", fontsize=12)
        symbols = ["^-", "o-"]
        for i, readers in enumerate(all_readers):
            lams, mags = zip(*[(reader.lam(), reader.magnetization()) for reader in readers])
            plt.plot(lams, mags, symbols[i], markersize=5, linewidth=0.5, label=folders[i])
        #plt.xlim(2.95, 3.15)
        #plt.ylim(-0.0025, 0.45)
        #plt.xticks([2.95, 3.0, 3.05, 3.1, 3.15])
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
            plt.plot(lams, energies, "v-", markersize=5, linewidth=0.5, label=folders[i])

    if kwargs["correlation_length"]:
        plt.ylabel(r"$\xi$")
        plt.xlabel(r"$\lambda$")
        for i, readers in enumerate(all_readers):
            lams, xis = zip(*[(reader.lam(), reader.correlation()) for reader in readers])
            plt.plot(lams, xis, "v-", color="C0", markersize=5, linewidth=0.5, label=folders[i])
        plt.xlim(2.7, 3.3)
        plt.grid(linestyle='--', linewidth=0.5)

    if kwargs["energy_chi"]:
        plt.ylabel(r"$\log|E-E_0|$")
        plt.xlabel(r"$1/\chi$")
        for i, readers in enumerate(all_readers):
            data = [(1/reader.ipeps.args["chi"], reader.energy()) for reader in readers if "chi" in reader.file]
            data.sort(reverse=True)
            inv_chis, energies = zip(*data)
            energies = np.array(energies) + 0.6689673179389798
            width,line,color =(0.5,"-","k") if "seed" in folders[i] else (0.7,"v-","C0")
            plt.plot(inv_chis, energies, line, markersize=4,color=color, linewidth=width)
        plt.grid(linestyle='--', linewidth=0.35)
        plt.yscale("log")
        plt.legend(["No gauge", "Random invertible gauges"])

    if kwargs["gradient"]:
        plt.ylabel(r"$\log|E-E_0|$")
        plt.xlabel(r"Epoch")
        for file in kwargs["gradient"].split(","):
            reader = Reader(os.path.join("data", folder, file))
            losses = np.array(reader.losses())
            if file[2] == "5":
                E0 = -0.6694037758828534
                color = "C2"
            if file[2] == "4":
                E0 = -0.6689670979898978 
                color = "C1"
            if file[2] == "3":
                E0 = -0.6681273941483516
                color = "C0"

           
            line = "-"  if len(file) > 6 else "--"

            #losses = abs(losses - E0) 
            plt.plot(range(len(losses)), losses, line, color=color, linewidth=1, label=file)
        #plt.ylim( -0.4911, -0.4909)
        #plt.xlim(60, 142)
        #plt.ylim(10**(-10), 10**(0))
        #plt.xticks(range(0, len(losses) + 1, 2))
        #plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        #plt.yscale("log")
        plt.grid(linestyle='--', linewidth=0.35, which='both')
        plt.legend()

    if kwargs["final_energies"]:
        heatmap_data = np.zeros((41, 17))
        for i, readers in enumerate(all_readers):
            final_energies, steps = [], []
            warmup_steps = [int(reader.ipeps.args["warmup_steps"]/5) for reader in readers] 
            gradient_steps = [reader.ipeps.args["Niter"] for reader in readers]
            final_energies = [reader.losses()[-1] for reader in readers]
            final_energies = abs(np.array(final_energies) + 0.4948685190401174)
            chi = readers[0].ipeps.args["chi"]

            for warmup, gradient, energy in zip(warmup_steps, gradient_steps, final_energies):
                heatmap_data[warmup, gradient] = energy
        plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='YlOrRd', norm=matplotlib.colors.LogNorm(vmax=10**(-2), vmin=10**(-3)))
        cbar = plt.colorbar(label=r"$|E - E_0|$")
        cbar.ax.tick_params(which='minor', labelsize=0)
        cbar.ax.yaxis.label.set_size(14)  # Increase the label size
        plt.xlabel(r"$N_g$", fontsize=14)
        plt.ylabel(r"$N_w$", fontsize=14)
        #plt.title(r"Heatmap of final energies for iPEPS states ($D = 4, \chi =10$)")
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
        plt.xlim(0.5, 10.5)
        plt.ylim(-0.5, 8.5)
        plt.gca().xaxis.set_minor_locator(plt.NullLocator())
        plt.gca().yaxis.set_minor_locator(plt.NullLocator())
        plt.yticks(ticks= range(9), labels = [0, 5, 10, 15, 20, 25, 30, 35, 40])


    if kwargs["gradient_norm"]:
        plt.ylabel(r"$\| \nabla E \|$")
        plt.xlabel(r"Epoch")
        for file in kwargs["gradient_norm"].split(","):
            reader = Reader(os.path.join("data", folder, file))
            norms = reader.gradient_norms()
            label = os.path.basename(reader.file).split('.')[0]
            plt.plot(range(len(norms)), norms, "v-", markersize=4, linewidth=0.5, label=label)

    if kwargs["ctmsteps"]:
        plt.ylabel(r"CTM Steps")
        plt.xlabel(r"Epoch")
        for file in kwargs["ctmsteps"].split(","):
            reader = Reader(os.path.join("data", folder, file))
            ctm_steps = reader.ctm_steps()
            label = os.path.basename(reader.file).split('.')[0]
            plt.plot(range(len(ctm_steps)), ctm_steps, "v-", markersize=4, linewidth=0.5, label=label)
        plt.xlim(0, 110)
        plt.grid(linestyle='--', linewidth=0.35)
        plt.gca().yaxis.get_major_locator().set_params(integer=True)
        #plt.ylim(5, 31)

    if kwargs["epochs"]:
        for i, readers in enumerate(all_readers):
            data = [(reader.ipeps.args["Niter"], len(reader.losses())) for reader in readers if "Niter" in reader.file]
            data.sort()
            steps, epochs = zip(*data)
            chi = readers[0].ipeps.args["chi"]
            plt.plot(steps, epochs, "v-", markersize=4, linewidth=0.5, label = rf"$ \chi = {chi} $")
        plt.ylabel(r"N")
        plt.xlabel(r"$N_g$")
        plt.grid(linestyle='--', linewidth=0.35)
        plt.legend()

    if kwargs["warmup_steps"]:
        for i, readers in enumerate(all_readers):
            data = [(reader.ipeps.args["warmup_steps"], reader.losses()[-1]) for reader in readers if "warmup_steps" in reader.file]
            data.sort()
            steps, energies = zip(*data)
            energies = np.array(energies) +0.4948685190401174
            chi = readers[-1].ipeps.args["chi"] 
            plt.plot(steps, energies, "v-", markersize=4.5, linewidth=1.3, label = rf"$ \chi = {chi} $")
        plt.ylabel(r"$\log|E - E_0|$", fontsize=14)
        plt.xlabel(r"$N_w$", fontsize=14)
        plt.grid(linestyle='--', linewidth=0.35)
        plt.legend(fontsize=13)
        plt.yscale("log")
        #plt.xlim(-1, 41)
        #plt.ylim(10**(-3.7), 10**(-2))
  
    plt.tight_layout()
    #plt.legend()
    #plt.savefig("figures/general_D5_convergence_gauge_comparison.png")
    plt.show()


@data.command()
@click.argument("old", type=click.Path())
@click.argument("new", type=click.Path())
@click.option("-s", "--server", is_flag=True, default=False, help="Wether to rename the folder/filename on the server.")
def rename(old: click.Path, new: click.Path, server: bool):
    """
    Rename a folder or filename to a new name.

    OLD is the current folder name. NEW is the new folder name.
    """
    args = read_cli_config()
    if server:
        with Connection(args["server_address"]) as c:
            c.run(f"cd PEPSFlow && git restore . && git pull", hide=True)
            c.run(f"cd PEPSFlow && source .venv/bin/activate && pepsflow data rename {old} {new}")
    else:
        old_path = os.path.join(args["data"], old)
        new_path = os.path.join(args["data"], new)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"\nRenamed [red]{old}[/] to [green]{new}")
        else:
            print(f"\n[red]Error:[/] The path [red]{old}[/] does not exist.")
            
        


@data.command()
@click.argument("path", type=click.Path())
@click.option("-s", "--server", is_flag=True, default=False, help="Wether to remove the file or directory from the server.")
def remove(path: click.Path, server: bool):
    """
    Remove a file or folder from the data directory.

    PATH is the file or folder to remove.
    """
    if server:
        args = read_cli_config()
        with Connection(args["server_address"]) as c:
            c.run(f"cd PEPSFlow && git restore . && git pull", hide=True)
            c.run(f"cd PEPSFlow && source .venv/bin/activate && pepsflow data remove {path}")   
            
    else: 
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
@click.option("-s", "--server", is_flag=True, default=False, help="Wether to show the contents of the folder on the server.")
@click.option("-f", "--file", default=None, type=click.Path(), help="File containing data, if not specified, all files in the folder are printed.")
@click.option("-st", "--state", is_flag=True, default=False, help="Print the iPEPS state.")
@click.option("-e", "--energy", is_flag=True, default=False, help="Print the energy.")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Print the magnetization.")
@click.option("-xi", "--correlation", is_flag=True, default=False, help="Print the correlation.")
@click.option("-o", "--losses", is_flag=True, default=False, help="Print the losses.")
@click.option("-p", "--params", is_flag=True, default=False, help="Print the parameters of the iPEPS model.")
@click.option("-c", "--ctmsteps", is_flag=True, default=False, help="Print the number of CTM warmup steps each epoch.")
def info(folder: click.Path, server: bool, **kwargs):
    """
    Print the information of the iPEPS models in the specified FOLDER.

    FOLDER contains the iPEPS models.
    """
    if server:
        args = read_cli_config()
        with Connection(args["server_address"]) as c:
            c.run(f"cd PEPSFlow && git restore . && git pull", hide=True)
            c.run(f"cd PEPSFlow && source .venv/bin/activate && pepsflow data info {folder} {' '.join([f'--{key}' for key, value in kwargs.items() if value])}")
    else:
        table = Table(title=f"iPEPS Information for Folder: {folder}", box=box.ASCII_DOUBLE_HEAD)
        
        table.add_column("Filename", justify="right", no_wrap=True, style="blue bold")
        if kwargs["energy"]: table.add_column("Energy", justify="right")
        if kwargs["magnetization"]: table.add_column("Magnetization", justify="right")
        if kwargs["correlation"]: table.add_column("Correlation", justify="right")
        if kwargs["losses"]: table.add_column("Losses", justify="left")
        if kwargs["state"]: table.add_column("State", justify="left")
        if kwargs["params"]: table.add_column("iPEPS Parameters", justify="left")
        if kwargs["ctmsteps"]: table.add_column("CTM Steps", justify="left")

        filenames = [kwargs["file"]] if kwargs["file"] else os.listdir(os.path.join("data", folder))
        
        for i, f in enumerate(filenames):
            if not f.endswith(".pth"): continue
            reader = Reader(os.path.join("data", folder, f))
            row = [f]
            if kwargs["energy"]: row.append(f"{reader.energy()}")
            if kwargs["magnetization"]: row.append(f"{reader.magnetization()}")
            if kwargs["correlation"]: row.append(f"{reader.correlation()}")
            if kwargs["losses"]: row.append(f"{reader.losses()}")
            if kwargs["state"]: row.append(f"{reader.iPEPS_state()}")
            if kwargs["params"]: row.append(f"{reader.ipeps.args}")
            if kwargs["ctmsteps"]: row.append(f"{reader.ctm_steps()}")
            style = "grey50" if i % 2 != 0 else "grey78"
            table.add_row(*row, style=style)

        print(table)
