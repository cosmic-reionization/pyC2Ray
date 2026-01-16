#include "utils.cuh"

#include <exception>
#include <iostream>
#include <sstream>

namespace asora {

    void safe_cuda(cudaError_t err, const std::source_location &loc) {
        if (err != cudaSuccess) {
            std::stringstream msg;
            msg << "CUDA Error " << cudaGetErrorName(err) << ": "
                << cudaGetErrorString(err) << ". At " << loc.function_name() << " in "
                << loc.file_name() << ":" << loc.line();
            std::cerr << msg.str() << "\n";
            throw std::runtime_error(msg.str());
        }
    }

}  // namespace asora
