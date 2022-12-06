#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// requires CUDA >= 10 and ARCH >= 70
// this is very slow compared to float or __half2, do not use!
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


// backward backward

template <typename scalar_t>
__global__ void kernel_grid_backward_backward(
    const scalar_t * __restrict__ grad_grad_grid,   // NxOHxOWx2
    const scalar_t * __restrict__ grad_outputs,     // NxCxOHxOW
    const scalar_t * __restrict__ features,         // NxCxIHxIW
    const scalar_t * __restrict__ grid,             // NxOHxOWx2
    scalar_t * __restrict__ grad2_grad,             // NxCxOHxOW
    scalar_t * __restrict__ grad2_features,         // NxCxIHxIW
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b>= N*OH*OW) return;

    // obtain index of the thread
    const uint32_t n = b / (OH * OW);
    const uint32_t oh = (b / OW) % OH;
    const uint32_t ow = b % OW;

    // skip to the corresponding grids
    grad_grad_grid = grad_grad_grid + (n * OH * OW + oh* OW  + ow) * 2;
    grid = grid + (n * OH * OW + oh* OW  + ow) * 2;

    grad_outputs = grad_outputs + (n * C * OH * OW) + oh * OW + ow;
    grad2_grad = grad2_grad + (n * C * OH * OW) + oh * OW + ow;

    features = features + n * C * IH * IW;
    grad2_features = grad2_features + n * C * IH * IW;

    const scalar_t ggx = grad_grad_grid[0];
    const scalar_t ggy = grad_grad_grid[1];
    const scalar_t gx = (grid[0] + 1) / 2.0 * (IW - 1);
    const scalar_t gy = (grid[1] + 1) / 2.0 * (IH - 1);
    if((gx < 0) || (gx > (IW - 1)) || (gy < 0) || (gy > (IH - 1)) || ((ggx == 0) && (ggy == 0))){
        return;
    }

    const uint32_t x0 = (uint32_t)floor(gx);
    const uint32_t y0 = (uint32_t)floor(gy);
    const uint32_t x1 = (uint32_t)ceil(gx);
    const uint32_t y1 = (uint32_t)ceil(gy);
    const scalar_t wx = (gx - (scalar_t)x0);
    const scalar_t wy = (gy - (scalar_t)y0);

    const scalar_t sx = (scalar_t)(IW - 1) / 2.0 * ggx;
    const scalar_t sy = (scalar_t)(IH - 1) / 2.0 * ggy;

    // pre-compute values
    const uint32_t c00 = y0 * IW + x0;
    const uint32_t c01 = y0 * IW + x1;
    const uint32_t c10 = y1 * IW + x0;
    const uint32_t c11 = y1 * IW + x1;

    for(uint32_t c = 0; c < C; c++){
        const uint32_t o00  = c * IH * IW + c00;
        const uint32_t o01  = c * IH * IW + c01;
        const uint32_t o10  = c * IH * IW + c10;
        const uint32_t o11  = c * IH * IW + c11;
        const scalar_t go = (grad_outputs + c * OH * OW)[0];

        atomicAdd(grad2_features + o00, go * (-(1 - wy) * sx - (1 - wx) * sy));
        atomicAdd(grad2_features + o01, go * ((1 - wy) * sx - wx * sy));
        atomicAdd(grad2_features + o10, go * (-wy * sx + (1 - wx) * sy));
        atomicAdd(grad2_features + o11, go * (wy * sx + wx * sy));

        // compute dy_dx
        const scalar_t f00 = features[o00];
        const scalar_t f01 = features[o01];
        const scalar_t f10 = features[o10];
        const scalar_t f11 = features[o11];

        const scalar_t g0x = -f00 * (1 - wy) + f01 * (1 - wy) - f10 * (wy)     + f11 * wy;
        const scalar_t g0y = -f00 * (1 - wx) - f01 * wx       + f10 * (1 - wx) + f11 * wx;

        const scalar_t dgx = g0x * sx;
        const scalar_t dgy = g0y * sy;

        grad2_grad[c * OH * OW] = dgx + dgy;
    }
}

template <typename scalar_t>
void grid_backward_backward_cuda(
    const scalar_t *grad_grad_grids, 
    const scalar_t *grad_outputs, 
    const scalar_t *features, 
    const scalar_t *grid, 
    scalar_t * grad2_grad, 
    scalar_t * grad2_feats, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW) {

    static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks_hashgrid = { div_round_up(N * OH * OW, N_THREAD), 1, 1 };
    kernel_grid_backward_backward<scalar_t><<<blocks_hashgrid, N_THREAD>>>(
        grad_grad_grids,
        grad_outputs,
        features,
        grid,
        grad2_grad,
        grad2_feats,
        N, C, IH, IW, OH, OW
    ); 
}

void grid_backward_backward(
    const at::Tensor grad_grad_grid, 
    const at::Tensor grad_outputs, 
    const at::Tensor features, 
    const at::Tensor grid, 
    at::Tensor grad2_grad, 
    at::Tensor grad2_features, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW
){

    CHECK_CUDA(grad_grad_grid);
    CHECK_CUDA(grad_outputs);
    CHECK_CUDA(features);
    CHECK_CUDA(grid);
    CHECK_CUDA(grad2_grad);
    CHECK_CUDA(grad2_features);

    CHECK_CONTIGUOUS(grad_grad_grid);
    CHECK_CONTIGUOUS(grad_outputs);
    CHECK_CONTIGUOUS(features);
    CHECK_CONTIGUOUS(grid);
    CHECK_CONTIGUOUS(grad2_grad);
    CHECK_CONTIGUOUS(grad2_features);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_grad_grid.scalar_type(), "grid_backward_backward", (
            [&] {
                grid_backward_backward_cuda<scalar_t>(
                    grad_grad_grid.data_ptr<scalar_t>(), 
                    grad_outputs.data_ptr<scalar_t>(), 
                    features.data_ptr<scalar_t>(), 
                    grid.data_ptr<scalar_t>(), 
                    grad2_grad.data_ptr<scalar_t>(), 
                    grad2_features.data_ptr<scalar_t>(), 
                    N, C, IH, IW, OH, OW
                );
            }
        )
    );
}
