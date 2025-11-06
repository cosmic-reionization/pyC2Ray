#pragma once

#include <cuda/std/array>
#include <cuda/std/utility>

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(
        double R, double *coldensh_out, double sig, double dr, double *ndens,
        double *xh_av, double *phi_ion, int num_src, int m1, double minlogtau,
        double dlogtau, int num_tau
    );

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(
        double Rmax_LLS, int q, int ns_start, int num_src_par, int num_src,
        int *src_pos, double *src_flux, double *coldensh_out, double sig, double dr,
        const double *ndens, const double *xh_av, double *phi_ion, int m1,
        const double *photo_thin_table, const double *photo_thick_table,
        double minlogtau, double dlogtau, int num_tau, int last_l, int last_r
    );

    // Short-characteristics interpolation function
    __device__ cuda::std::pair<double, double> cinterp_gpu(
        int i, int j, int k, int i0, int j0, int k0, const double *coldensh_out,
        double sigma_HI_at_ion_freq, int m1
    );

    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int s, int q);

}  // namespace asora
