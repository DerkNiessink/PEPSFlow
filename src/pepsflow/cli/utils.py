import click
import os
import pathlib
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape
import json

from pepsflow.iPEPS.reader import iPEPSReader


def get_observables(
    folder: str,
    magnetization: bool,
    energy: bool,
    correlation_length: bool,
    gradient: click.Path,
    gradient_norm: click.Path,
    energy_convergence: click.Path,
    energy_chi: click.Path,
) -> tuple[list, list, list, list, list]:
    """
    Get the observables of all iPEPS models in the specified folder.

    Args:
        folder (str): Folder containing the iPEPS models.
        magnetization (bool): Compute the magnetization.
        energy (bool): Compute the energy.
        correlation_length (bool): Compute the correlation length.
        gradient (str): Desired file to plot the gradient.
        gradient_norm: (str): Desired file to plot the gradient norm.
        energy_convergence (str): Desired file to plot the energy convergence.

    Returns:
        dict: Dictionary containing the observables, keys: "lam", "M", "E", "xi", "losses", "norms", "energy_convergence"
    """
    data = {"lam": [], "M": [], "E": [], "xi": []}

    # FOR ALL FILES IN THE FOLDER
    for file in os.listdir(os.path.join("data", folder)):
        if not file.endswith(".pth"):
            continue
        reader = iPEPSReader(os.path.join("data", folder, file))
        data["lam"].append(reader.lam())

        if magnetization:
            data["M"].append(reader.magnetization())
        if energy:
            data["E"].append(reader.energy())
        if correlation_length:
            data["xi"].append(reader.correlation())

    # FILE SPECIFIC DATA
    if gradient:
        reader = iPEPSReader(os.path.join("data", folder, gradient))
        data["losses"] = reader.losses()

    if gradient_norm:
        reader = iPEPSReader(os.path.join("data", folder, gradient_norm))
        data["norms"] = reader.gradient_norms()

    if energy_convergence or energy_chi:
        file = energy_chi if energy_chi else energy_convergence
        with open(os.path.join("data", folder, file)) as f:
            data["energy_convergence"] = json.load(f)

    return data


def walk_directory(directory: pathlib.Path, tree: Tree, concise: bool) -> None:
    """
    Recursively build a Tree with directory contents.

    Args:
        directory (pathlib.Path): Directory to walk.
        tree (Tree): Tree to add to.
        concise (bool): If True, do not show the files in the directory.
    """

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
