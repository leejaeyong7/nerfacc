#ifndef _GRID_ENCODE_H
#define _GRID_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

void grid_backward_backward(
    const at::Tensor grad_grad_grids, 
    const at::Tensor grad_outputs, 
    const at::Tensor features, 
    const at::Tensor grids, 
    at::Tensor grad2_grad, 
    at::Tensor grad2_features, 
    const uint32_t N, const uint32_t C, 
    const uint32_t IH, const uint32_t IW, 
    const uint32_t OH, const uint32_t OW);

#endif