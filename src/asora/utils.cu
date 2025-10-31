#include "utils.cuh"

#include <exception>
#include <format>
#include <iostream>

namespace asora {

    void safe_cuda(cudaError_t err, const std::source_location &loc) {
        if (err != cudaSuccess) {
            auto msg = std::format(
                "CUDA Error {}: {}. At {} in {}:{}", cudaGetErrorName(err),
                cudaGetErrorString(err), loc.function_name(), loc.file_name(),
                loc.line()
            );
            std::cerr << msg << "\n";
            throw std::runtime_error(msg);
        }
    }

}  // namespace asora
