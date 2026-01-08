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

    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s) {
        if (s == 0) return {q, 0, 0};

        auto s_top = 2 * q * (q + 1) + 1;
        if (s == s_top) return {q - 1, 0, -1};

        int sign = 1;
        int q1 = q;
        if (s > s_top) {
            s -= s_top;
            q1 -= 1;
            sign = -1;
        }

        auto j = (s - 1) / (2 * q1);
        auto i = (s - 1) % (2 * q1) + j - q1;
        if (i + j > q1) {
            i -= q1;
            j -= q1 + 1;
        }

        return {i, j, sign * (q - abs(i) - abs(j))};
    }

    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k) {
        auto q = abs(i) + abs(j) + abs(k);
        if (i == q && j == 0 && k == 0) return {q, 0};

        auto s_top = 2 * q * (q + 1) + 1;
        if (i == q - 1 && j == 0 && k == -1) return {q, s_top};

        auto q1 = k >= 0 ? q : q - 1;
        auto s_off = k >= 0 ? 0 : s_top;

        // Guess a solution
        auto s0 = (2 * q1 - 1) * j + i + q1 + 1;
        if (s0 > 0) {
            auto j0 = (s0 - 1) / (2 * q1);
            auto i0 = (s0 - 1) % (2 * q1) + j - q1;
            if (i0 + j0 > q1) {
                i0 -= q1;
                j0 -= q1 + 1;
            }
            if (i0 == i && j0 == j) return {q, s0 + s_off};
        }

        // It's the other solution
        s0 += 2 * q1 * (q1 + 1) - 1 + s_off;
        return {q, s0};
    }

}  // namespace asora
