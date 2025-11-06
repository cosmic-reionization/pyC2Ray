#include "tests.cuh"

#include "raytracing.cuh"
#include "utils.cuh"

#include <iostream>

namespace asoratest {

    namespace {

        __global__ void cinterp_gpu_kernel(
            double *out_data, const double *dens_data, int3 pos0, int m1
        ) {
            auto &[i, j, k] = threadIdx;
            auto &[i0, j0, k0] = pos0;

            auto &&[cdens, path] = asora::cinterp_gpu(
                i + i0, j + j0, k + k0, i0, j0, k0, dens_data, 1.0, m1
            );

            auto idx = k + blockDim.z * (j + blockDim.y * i);
            out_data[2 * idx] = cdens;
            out_data[2 * idx + 1] = path;
        }

    }  // namespace

    // Arrays are host pointers:
    void cinterp_gpu(
        double *out_data, const std::array<size_t, 4> &out_shape,
        const double *dens_data, size_t dens_size, const int3 &pos0, int m1
    ) {
        double *dens_dev;
        asora::safe_cuda(cudaMalloc(&dens_dev, dens_size));
        asora::safe_cuda(
            cudaMemcpy(dens_dev, dens_data, dens_size, cudaMemcpyHostToDevice)
        );

        size_t out_size = sizeof(double);
        for (auto &dim : out_shape) out_size *= dim;
        double *out_dev;
        asora::safe_cuda(cudaMalloc(&out_dev, out_size));

        uint3 ts = {
            static_cast<unsigned int>(out_shape[0]),  //
            static_cast<unsigned int>(out_shape[1]),  //
            static_cast<unsigned int>(out_shape[2])   //
        };
        cinterp_gpu_kernel<<<1, ts>>>(out_dev, dens_dev, pos0, m1);

        asora::safe_cuda(cudaPeekAtLastError());
        asora::safe_cuda(
            cudaMemcpy(out_data, out_dev, out_size, cudaMemcpyDeviceToHost)
        );

        asora::safe_cuda(cudaFree(out_dev));
        asora::safe_cuda(cudaFree(dens_dev));
    }

    std::array<int, 3> linthrd2cart(int s, int q) {
        auto [i, j, k] = asora::linthrd2cart(s, q);
        return {i, j, k};
    }

}  // namespace asoratest
