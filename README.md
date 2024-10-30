# PEPSFlow

PEPSFlow is a Python package for the simulation of PEPS (Projected Entangled Pair States) in 2D using automatic differentiation.

## Program flow and structure

<figure>
    <img src="assets/pepsflow.svg" />
    <figcaption><em>Figure 1: PEPSFlow program flow </em> </figcaption>
</figure>

<br>
<br>
<br>
See Figure 1 for a dependency graph of PEPSFlow. The modules are explained below:

#### `train.iPEPS`
Contains the class `iPEPS` which is derived from the `torch.nn.Module` class. This class represents an iPEPS state, so it has the iPEPS tensor ("A tensor") as trainable parameters and attributes like the corner and edge tensor obtained from the CTM algorithm. The `forward` method of this class executes one CTM step and computes the energy using the new corner and edge tensors. After optimization of the parameters this class is saved in the `data` folder as a `.pth` file.
#### `train.iPEPS_trainer`
Contains the class `iPEPSTrainer`. This class is used for optimizing the parameters and saving the resulting `train.iPEPS` instance. This class takes a dictionary of optimization parameters as argument.
#### `train.iPEPS_reader`
Contains the class `iPEPSReader`, which is used for reading the `train.iPEPS` instances in `.pth` files.
