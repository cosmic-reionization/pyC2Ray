#pragma once

#include <cuda/std/utility>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, double sig, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size = 256
    );

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        double Rmax_LLS, int q, size_t ns_start, size_t num_src, int *src_pos,
        double *src_flux, double *coldensh_out, double sig, double dr,
        const double *ndens, const double *xh_av, double *phi_ion, size_t m1,
        const double *photo_thin_table, const double *photo_thick_table,
        double minlogtau, double dlogtau, size_t num_tau, int last_l, int last_r
    );

    // Short-characteristics interpolation function
    __device__ cuda::std::pair<double, double> cinterp_gpu(
        int i, int j, int k, int i0, int j0, int k0, const double *coldensh_out,
        double sigma_HI_at_ion_freq, size_t m1
    );

}  // namespace asora
