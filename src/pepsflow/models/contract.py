import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import opt_einsum


class BlockedEinsum(nn.Module):
    def __init__(self, equation: str, block_dim: str, batch_size: int, use_checkpoint: bool = False):
        super().__init__()
        self.equation = equation
        self.block_dim = block_dim
        self.batch_size = batch_size
        self.use_checkpoint = use_checkpoint

        # Parse equation
        if "->" not in equation:
            raise ValueError("Only explicit einsum equations with '->' are supported.")
        input_part, output_subscript = equation.split("->")
        self.input_subscripts = [s.strip() for s in input_part.split(",")]
        self.output_subscript = output_subscript.strip()

        # Find which tensor/dim to block
        found = False
        for tidx, sub in enumerate(self.input_subscripts):
            if block_dim in sub:
                self.tidx = tidx
                self.didx = sub.index(block_dim)
                found = True
                break
        assert found, f"block_dim '{block_dim}' not found in any input subscript."
        assert (
            block_dim in self.output_subscript
        ), f"block_dim '{block_dim}' must appear in output for meaningful slicing."
        count = sum([sub.count(block_dim) for sub in self.input_subscripts])
        assert count == 1, f"block_dim '{block_dim}' is contracted (appears in multiple inputs), cannot block."
        self.out_dim = self.output_subscript.index(block_dim)

    def _forward_impl(self, *tensors):
        tensor_to_block = tensors[self.tidx]
        chunks = torch.split(tensor_to_block, self.batch_size, dim=self.didx)
        results = []
        for chunk in chunks:
            chunk_tensors = list(tensors)
            chunk_tensors[self.tidx] = chunk
            chunk_result = opt_einsum.contract(self.equation, *chunk_tensors)
            results.append(chunk_result)
        return torch.cat(results, dim=self.out_dim)

    def forward(self, *tensors):
        if self.use_checkpoint:
            return checkpoint(self._forward_impl, *tensors, use_reentrant=False)
        else:
            return self._forward_impl(*tensors)


# ========== Complex Example usage ==========
if __name__ == "__main__":

    # abcd, ce, df, ag -> ebfg
    # Let's set shapes:
    # a=16, b=8, c=12, d=10, e=6, f=7, g=9
    A = torch.randn(16, 8, 12, 10, requires_grad=True, dtype=torch.float64)  # abcd
    B = torch.randn(12, 6, requires_grad=True, dtype=torch.float64)  # ce
    C = torch.randn(10, 7, requires_grad=True, dtype=torch.float64)  # df
    D = torch.randn(16, 9, requires_grad=True, dtype=torch.float64)  # ag

    equation = "abcd,ce,df,ag->ebfg"
    block_dim = "b"
    batch_size = 3

    # Reference
    ref = opt_einsum.contract(equation, A, B, C, D)

    # Blocked
    model = BlockedEinsum(equation, block_dim=block_dim, batch_size=batch_size, use_checkpoint=True)
    res = model(A, B, C, D)

    print("Relative error:", (ref - res).norm() / ref.norm())
    assert torch.allclose(ref, res, atol=1e-6)

    # Test backward
    grad = torch.randn_like(ref)
    ref.backward(grad)
    gradA_ref = A.grad.clone()
    gradB_ref = B.grad.clone()
    gradC_ref = C.grad.clone()
    gradD_ref = D.grad.clone()
    A.grad.zero_()
    B.grad.zero_()
    C.grad.zero_()
    D.grad.zero_()
    res.backward(grad)
    print("Backward gradA error:", (A.grad - gradA_ref).norm() / gradA_ref.norm())
    print("Backward gradB error:", (B.grad - gradB_ref).norm() / gradB_ref.norm())
    print("Backward gradC error:", (C.grad - gradC_ref).norm() / gradC_ref.norm())
    print("Backward gradD error:", (D.grad - gradD_ref).norm() / gradD_ref.norm())
    assert torch.allclose(A.grad, gradA_ref, atol=1e-12)
    assert torch.allclose(B.grad, gradB_ref, atol=1e-12)
    assert torch.allclose(C.grad, gradC_ref, atol=1e-12)
    assert torch.allclose(D.grad, gradD_ref, atol=1e-12)
    print("All tests passed!")
