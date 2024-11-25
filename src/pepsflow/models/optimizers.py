import torch


class Optimizer:
    """
    Class to define the optimizers used in the training process.
    """

    def __init__(self, optimizer: str, params: torch.Tensor, **kwargs: dict):
        match optimizer:
            case "adam":
                self.optimizer = torch.optim.Adam(params, kwargs["lr"])
            case "lbfgs":
                self.optimizer = torch.optim.LBFGS(params, kwargs["lr"], 1, line_search_fn=kwargs["line_search_fn"])
            case _:
                raise ValueError(f"Optimizer {optimizer} not recognized.")

    def step(self, closure: callable) -> None:
        """
        Perform a single optimization step.

        Args:
            closure (callable): Function to evaluate the model.

        Returns:
            Loss value.
        """
        return self.optimizer.step(closure)

    def zero_grad(self) -> None:
        """
        Zero the gradients of the parameters.
        """
        self.optimizer.zero_grad()
