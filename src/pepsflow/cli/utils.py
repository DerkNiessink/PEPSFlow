import pathlib
from rich.tree import Tree
from rich.text import Text
from rich.filesize import decimal
from rich.markup import escape
import configparser
import ast


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
            # style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(f"{escape(path.name)}")
            walk_directory(path, branch, concise)
        elif not concise:
            text_filename = Text(path.name)
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            tree.add(text_filename)


def read_cli_config() -> dict:
    """
    Read the command line interface parameters from the configuration file.

    Returns:
        dict: Dictionary containing command line interface parameters.
    """
    parser = configparser.ConfigParser()
    parser.optionxform = lambda option: option  # Preserve the case of the keys
    parser.read("src/pepsflow/pepsflow.cfg")
    section = "parameters.cli"
    args = {}
    for key, value in parser.items(section):
        try:
            args[key] = ast.literal_eval(value)
        except ValueError:
            pass
    return args
