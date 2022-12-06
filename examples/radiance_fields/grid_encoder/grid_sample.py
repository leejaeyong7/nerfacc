import torch
from .grid_backward import grid_backward

def grid_sample(input, grid):
    return grid_sample_cuda.apply(input, grid)

class grid_sample_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', align_corners=True)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = grid_backward.apply(grad_output, input, grid)
        return grad_input, grad_grid
