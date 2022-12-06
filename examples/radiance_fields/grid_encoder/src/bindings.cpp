#include <torch/extension.h>

#include "grid_backward.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_backward_backward", &grid_backward_backward, "grid backward (CUDA)");
}