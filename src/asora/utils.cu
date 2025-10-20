#include "utils.cuh"

#include <exception>
#include <format>

namespace asora {

    void safe_cuda(cudaError_t err, const std::source_location &loc) {
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::format("CUDA Error {}: {}. At {} in {}:{}", cudaGetErrorName(err),
                            cudaGetErrorString(err), loc.function_name(),
                            loc.file_name(), loc.line()));
    }

}  // namespace asora
