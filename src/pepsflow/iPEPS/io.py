import torch
import os
import sys
import json
from rich import print
from typing import Any

from pepsflow.ipeps.ipeps import iPEPS, make_ipeps


class IO:
    """
    Class to handle the input and output of iPEPS models by serializing and
    deserializing to JSON files.
    """

    @classmethod
    def make_json_serializable(self, obj) -> Any:
        """Recursively convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: IO.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [IO.make_json_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj

    @classmethod
    def save(self, ipeps: iPEPS, fn: str) -> None:
        """Save the iPEPS model to a JSON file.

        Args:
            ipeps (iPEPS): The iPEPS model to save.
            fn (str): The filename to save the model to.
        """

        fn = fn + ".json" if not fn.endswith(".json") else fn
        folder = os.path.dirname(fn)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        data = {"ipeps_args": ipeps.args, "state": ipeps.params, "map": ipeps.map, "data": ipeps.data}
        data = IO.make_json_serializable(data)

        with open(fn, "w") as f:
            json.dump(data, f, indent=4)
        sys.stdout.flush()
        print(f"[green bold] \nData saved to {fn}")

    @classmethod
    def load(self, fn: str) -> iPEPS:
        """Load the iPEPS model from a JSON file.

        Args:
            fn (str): The filename to load the model from.
        """
        fn = fn + ".json" if not fn.endswith(".json") else fn
        with open(fn, "r") as f:
            data = json.load(f)

        ipeps_args = data["ipeps_args"]
        ipeps = make_ipeps(ipeps_args)
        ipeps.map = torch.tensor(data["map"]) if data["map"] is not None else None
        dtype = torch.float64 if ipeps_args["dtype"] == "double" else torch.float32
        ipeps.params = torch.nn.Parameter(torch.tensor(data["state"], dtype=dtype))
        ipeps.data = data["data"]

        # Convert the gauges to tensors
        gauges = ipeps.data.get("Gauges [U1, U2]", [torch.eye(ipeps_args["D"]), torch.eye(ipeps_args["D"])])
        ipeps.data["Gauges [U1, U2]"] = [torch.tensor(gauges[0], dtype=dtype), torch.tensor(gauges[1], dtype=dtype)]

        print(f"[green bold] \nData loaded from {fn}")
        return ipeps
