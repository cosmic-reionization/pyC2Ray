#pragma once

#include <cuda_runtime.h>

#include <source_location>

namespace asora {

    // Throw exception for CUDA errors.
    void safe_cuda(cudaError_t err,
                   const std::source_location &loc = std::source_location::current());

}  // namespace asora
