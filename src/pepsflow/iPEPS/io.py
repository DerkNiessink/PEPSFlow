import torch
import os
import sys
import json
from rich import print

from pepsflow.ipeps.ipeps import iPEPS, make_ipeps


class IO:
    """
    Class to handle the input and output of iPEPS models by serializing and
    deserializing to JSON files.

    Args:
        file (str): File containing the iPEPS model.
    """

    @classmethod
    def save(self, ipeps: iPEPS, fn: str) -> None:
        fn = fn + ".json" if not fn.endswith(".json") else fn
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        data = {
            "ipeps_args": ipeps.args,
            "state": ipeps.params.tolist(),
            "map": ipeps.map.tolist() if ipeps.map is not None else None,
            "data": ipeps.data,
        }
        with open(fn, "w") as f:
            json.dump(data, f, indent=4)
        sys.stdout.flush()
        print(f"[green bold] \nData saved to {fn}")

    @classmethod
    def load(self, fn: str) -> iPEPS:
        fn = fn + ".json" if not fn.endswith(".json") else fn
        with open(fn, "r") as f:
            data = json.load(f)

        ipeps_args = data["ipeps_args"]
        ipeps = make_ipeps(ipeps_args)
        ipeps.map = torch.tensor(data["map"]) if data["map"] is not None else None
        dtype = torch.float64 if ipeps_args["dtype"] == "double" else torch.float32
        ipeps.params = torch.nn.Parameter(torch.tensor(data["state"], dtype=dtype))
        ipeps.data = data["data"]
        print(f"[green bold] \nData loaded from {fn}")
        return ipeps
