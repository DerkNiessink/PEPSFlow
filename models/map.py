import numpy as np

from models.tensors import Tensors


def map_to_A_tensor(new_unique_values):
    tensor = Tensors.A_solution()
    original_unique_values = np.unique(tensor)

    if len(new_unique_values) != 12:
        raise ValueError("The new unique values list must contain exactly 12 elements.")

    value_map = {
        old_val: new_val
        for old_val, new_val in zip(original_unique_values, new_unique_values)
    }

    map_function = np.vectorize(lambda x: value_map[x])
    new_tensor = map_function(tensor)

    return new_tensor
