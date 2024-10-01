import torch
import numpy as np


class CustomSVD(torch.autograd.Function):
    """
    Custom SVD function to compute the singular value decomposition of a tensor.
    This function supportss backpropagation through the SVD operation.
    """

    @staticmethod
    def forward(self, A):
        """
        Forward pass of the custom SVD function. Computes the singular value
        decomposition of the input tensor.

        Args:
            A (torch.Tensor): Input tensor of shape (m, n).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: U, S, V matrices
            of the SVD decomposition
        """
        U, S, V = torch.linalg.svd(A, full_matrices=False)
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        """
        Backward pass of the custom SVD function. Computes the gradient of the
        input tensor with respect to the output tensor. This function is save
        for degenerate singular values.

        Args:
            dU (torch.Tensor): Gradient of the loss with respect to U.
            dS (torch.Tensor): Gradient of the loss with respect to S.
            dV (torch.Tensor): Gradient of the loss with respect to V.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input tensor.
        """
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        N_rows = U.size(0)
        N_columns = V.size(0)
        N_singular_values = len(S)

        # Off diagonal terms
        F = S - S[:, None]
        # Avoid division by zero
        F = F / (F**2 + 1e-12)
        F.diagonal().fill_(0)

        # Diagonal terms
        G = S + S[:, None]
        G.diagonal().fill_(np.inf)
        G = 1 / G

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F + G) * (UdU - UdU.t()) / 2
        Sv = (F - G) * (VdV - VdV.t()) / 2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt

        # fmt: off
        if N_rows > N_singular_values:
            dA += (torch.eye(N_rows, dtype=dU.dtype, device=dU.device) - U @ Ut) @ (dU / S) @ Vt
            
        if N_columns > N_singular_values:
            dA += (U / S) @ dV.t() @ (torch.eye(N_columns, dtype=dU.dtype, device=dU.device) - V @ Vt)
            
        return dA
