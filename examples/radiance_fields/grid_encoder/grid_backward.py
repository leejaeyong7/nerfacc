import torch
from torch.autograd import Function

from pkg_resources import parse_version
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a') # Allow prerelease builds of 1.11

from .backend import _backend

class grid_backward(Function):
    @staticmethod
    def forward(ctx, grad_output, features, grids):
        N, C, IH, IW = features.shape
        N, OH, OW, _ = grids.shape
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')[0]

        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_features, grad_grids = op(grad_output, features, grids, 0, 0, True, output_mask)
        else:
            grad_features, grad_grids = op(grad_output, features, grids, 0, 0, True)

        ctx.save_for_backward(grad_output, features, grids)
        ctx.dims = [N, C, IH, IW, OH, OW]

        return grad_features, grad_grids

    @staticmethod
    def backward(ctx, grad_grad_features, grad_grad_grids):
        grad_output, features, grids = ctx.saved_tensors
        N, C, IH, IW, OH, OW = ctx.dims

        if not grad_grad_grids.is_contiguous():
            grad_grad_grids = grad_grad_grids.contiguous()

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if not grids.is_contiguous():
            grids = grids.contiguous()

        # NxCxIHxIW
        grad_grad = torch.zeros_like(grad_output)

        # NxCxIHxIW
        grad2_features = torch.zeros_like(grad_grad_features)
        _backend.grid_backward_backward(grad_grad_grids, grad_output, features, grids, grad_grad, grad2_features, N, C, IH, IW, OH, OW)

        return grad_grad, grad2_features, None


