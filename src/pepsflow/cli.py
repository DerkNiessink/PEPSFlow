import click
import subprocess
import re
import matplotlib.pyplot as plt
import os

from pepsflow.train.iPEPS_reader import iPEPSReader

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
        for dirpath, dirnames, filenames in os.walk("data"):
            level = dirpath.replace("data", '').count(os.sep)
            indent = ' ' * 2 * level
            if folder is None or os.path.basename(dirpath) == folder:
                print(f'{indent}/{os.path.basename(dirpath)}')
                subindent = ' ' * 2 * (level + 1)
                if not concise or folder is not None:
                    for filename in filenames:
                        print(f'{subindent}{filename}')

@click.group(invoke_without_command=True)
@click.pass_context
def params(ctx):
    """
    Show the current parameters for the optimization of the iPEPS tensor network.
    """
    if ctx.invoked_subcommand is None:
        with open("src/pepsflow/run.sh", 'r') as file:
            lines = file.readlines()
        print("")
        in_param_section = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("# START PARAMS"):
                in_param_section = True
                continue
            elif stripped_line.startswith("# END PARAMS"):
                if in_param_section:
                    break
            if in_param_section:
                print(stripped_line)
        print("")

cli.add_command(data)
cli.add_command(params)

@params.command()
def optimize():
    """
    Optimize the iPEPS tensor network with the specified parameters.
    """
    # Fix the line endings in the run.sh file to Unix format
    run_file = "src/pepsflow/run.sh"
    with open(run_file, 'r', newline='') as f:
        script_content = f.read().replace('\r\n', '\n') 
    with open(run_file, 'w', newline='') as f:
        f.write(script_content)

    subprocess.run(["bash", run_file])


@params.command()
@click.option("--chi", type=int, help="Environment bond dimension in the CTMRG algorithm")
@click.option("--d", type=int, help="Bulk bond dimension of the iPEPS")
@click.option("-df","--data_folder", type=str, help="Folder containing iPEPS models")
@click.option("--gpu/--no-gpu", type=bool, help="Run the model on the GPU if available")
@click.option("--lam", type=float, multiple=True, help="Value(s) of the parameter lambda in the tranverse-field Ising model")
@click.option("--max_iter", type=int, help="Maximum number of iterations for the optimizer")
@click.option("--runs", type=int, help="Number of runs to train the model. Applies to random initialization. The program will choose the best model based on the lowest energy.")
@click.option("-lr","--learning_rate", type=float, help="Learning rate for the optimizer")
@click.option("--epochs", type=int, help="Maximum number of epochs to train the model")
@click.option("-per", "--perturbation", type=float, help="Amount of perturbation to apply to the initial state")
@click.option("-sf", "--save_folder", type=str, help="Folder to save the iPEPS model in.")
def set(**args):
    """
    Set specific parameters for the optimization of the iPEPS tensor network.
    """
    with open("src/pepsflow/run.sh", 'r') as f:
        script_content = f.read()

    # Regular expression pattern to match parameters
    for param, value in args.items():
        if value is not None and value != ():
            # Format for multiple values (like lam)
            if param == 'lam' and isinstance(value, tuple):  # Check if 'lam' has multiple values
                value_str = f"({ ' '.join(map(str, value)) })"
            elif isinstance(value, bool):
                value_str = 'true' if value else 'false'
            else:
                value_str = str(value)
    
            # Replace the existing parameter with the new value in the script content
            script_content = re.sub(f"^{param}=[^\\n]*", f"{param}={value_str}", script_content, flags=re.MULTILINE)


    with open("src/pepsflow/run.sh", 'w') as f:
        f.write(script_content)



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
            mag_ax.plot(lambda_values, reader.get_magnetizations(), "v-", markersize=4, linewidth=0.5, label=folder)

        if energy or plot_all:
            en_ax.plot(lambda_values, reader.get_energies(), "v-", markersize=4, linewidth=0.5, label=folder)

        if correlation_length or plot_all:
            xi_ax.plot(lambda_values, reader.get_correlations(), "v-", markersize=4, linewidth=0.5, label=folder)

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
        plt.figure(mag_figure)
        plt.show()

    if energy or plot_all:
        en_ax.legend()
        plt.figure(en_figure)
        plt.show()

    if correlation_length or plot_all:
        xi_ax.legend()
        plt.figure(xi_figure)
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
