#pragma once

// Single source for both GPU backends. The ASORA sources are written against
// the CUDA runtime API; when compiled with hipcc the cuda* names used here
// are mapped to their HIP equivalents.
//   - nvcc                              -> native CUDA
//   - hipcc, HIP_PLATFORM=amd (ROCm)    -> HIP on AMD GPUs
//   - hipcc, HIP_PLATFORM=nvidia        -> HIP on NVIDIA GPUs (via nvcc)
// Only add mappings for names that are actually used in the sources.

#if defined(__HIPCC__)

#include <hip/hip_runtime.h>

#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess

#else

#include <cuda_runtime.h>

#endif

// libcu++ (cuda::std) ships with the CUDA toolkit but not with ROCm. On AMD,
// hipcc (clang) allows the constexpr std:: utilities used here (pair, tie,
// array, swap) in device code, so cuda::std is aliased to std.
#if defined(__HIP_PLATFORM_AMD__)

#include <array>
#include <tuple>
#include <utility>

namespace cuda {
    namespace std = ::std;
}

#else

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#endif
