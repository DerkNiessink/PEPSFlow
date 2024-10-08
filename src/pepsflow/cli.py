import click
import subprocess
import re
import matplotlib.pyplot as plt
import os

from pepsflow.train.iPEPS_reader import iPEPSReader

# fmt: off

@click.group()
def cmd_group():
    pass


@cmd_group.command()
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

@cmd_group.command()
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


@cmd_group.command()
def params():
    """
    Show the current parameters for the optimization of the iPEPS tensor network.
    """
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

@cmd_group.command()
@click.argument("folder", type=str)
@click.option("-e", "--energy", is_flag=True, default=False, help="Plot the energy as a function of lambda")
@click.option("-m", "--magnetization", is_flag=True, default=False, help="Plot the magnetization as a function of lambda")
@click.option("-xi", "--correlation_length", is_flag=True, default=False, help="Plot the correlation length as a function of lambda")
def plot(folder: str, correlation_length: bool, energy: bool, magnetization: bool):
    """
    Plot the observables of the iPEPS models.
    """
    reader = iPEPSReader(os.path.join("data", folder))
    lambda_values = reader.get_lambdas()
    
    if magnetization:
        plt.figure(figsize=(6, 4))
        plt.plot(lambda_values, reader.get_magnetizations(), "v-", markersize=4, linewidth=0.5)
        plt.ylabel(r"$\langle M_z \rangle$")
        plt.xlabel(r"$\lambda$")
        plt.show()
    if energy:    
        plt.figure(figsize=(6, 4))
        plt.plot(lambda_values, reader.get_energies(), "v-", markersize=4, linewidth=0.5)  
        plt.ylabel(r"$E$")
        plt.xlabel(r"$\lambda$")
        plt.show()
    if correlation_length:
        plt.figure(figsize=(6, 4))
        plt.plot(lambda_values, reader.get_correlations(), "v-", markersize=4, linewidth=0.5)
        plt.ylabel(r"$\xi$")
        plt.xlabel(r"$\lambda$")
        plt.show()
    if not magnetization and not energy and not correlation_length:
        print("Please specify the observables to plot. See pepsflow plot --help for more information.")
