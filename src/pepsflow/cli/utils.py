import click
import os
import pathlib
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape

from pepsflow.train.iPEPS_reader import iPEPSReader


def get_observables(
    folder: str,
    magnetization: bool,
    energy: bool,
    correlation_length: bool,
    gradient: click.Path,
) -> tuple[list, list, list, list, list]:
    """
    Get the observables of all iPEPS models in the specified folder.

    Args:
        folder (str): Folder containing the iPEPS models.
        magnetization (bool): Compute the magnetization.
        energy (bool): Compute the energy.
        correlation_length (bool): Compute the correlation length.
        gradient (str): Desired file to plot the gradient.
    """
    magnetizations, energies, correlations, lambdas, losses = [], [], [], [], []
    for file in os.listdir(os.path.join("data", folder)):
        reader = iPEPSReader(os.path.join("data", folder, file))
        lambdas.append(reader.get_lam())
        if magnetization:
            magnetizations.append(reader.get_magnetization())
        if energy:
            energies.append(reader.get_energy())
        if correlation_length:
            correlations.append(reader.get_correlation())

    if gradient:
        reader = iPEPSReader(os.path.join("data", folder, gradient))
        losses = reader.get_losses()

    return lambdas, magnetizations, energies, correlations, losses


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
