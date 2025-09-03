<p align="center">
  <img src="https://github.com/user-attachments/assets/0ef5c5c2-3a1a-4644-a281-7d82d185699e" alt="Description" width="400"/>
</p>
 <h1 align="center">PEPSFlow</h1>


PEPSFlow is a Python package for the simulation of PEPS (Projected Entangled Pair States) in 2D using automatic differentiation.

## How to install

Requirements: [PyTorch 1.0+](https://pytorch.org/)

1. Clone this repository.
2. Create a virtual environment (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [venv](https://docs.python.org/3/library/venv.html)).
3. Use the package manager [poetry](https://python-poetry.org/) to intstall the dependencies:
   ```
   $\PEPSFlow> poetry install
   ```

## How to use

The pepsflow command interface (CLI) consists of two main subcommands:

* `pepsflow params`: handling of current parameters and optimization.
* `pepsflow data`: handling the saved data.

Add `--help` to a command to show more information, e.g:

```
$\PEPSFlow> pepsflow params --help
```
```
$\PEPSFlow> pepsflow data --help
```






