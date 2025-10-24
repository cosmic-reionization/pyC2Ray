#pragma once

// ========================================================================
// Header file for OCTA raytracing library.
// Functions defined and documented in raytracing_gpu.cu
// ========================================================================

// Modulo function with Fortran convention
/*
inline int modulo(const int &a, const int &b);
inline __device__ int modulo_gpu(const int &a, const int &b);

// Device sign function
inline __device__ int sign_gpu(const double &x);

// Flat array index from i,j,k coordinates
inline __device__ int mem_offst_gpu(const int &i, const int &j, const int &k, const int
&N);

// Mapping from linear thread space to the cartesian coords of a q-shell in
// asora
__device__ void linthrd2cart(const int &, const int &, int &, int &);
*/

namespace asora {

    // Raytrace all sources and compute photoionization rates
    void do_all_sources_gpu(double R, double *coldensh_out, double sig, double dr,
                            double *ndens, double *xh_av, double *phi_ion, int num_src,
                            int m1, double minlogtau, double dlogtau, int num_tau);

    // Raytracing kernel, called by do_all_sources
    __global__ void evolve0D_gpu(double Rmax_LLS, int q, int ns_start, int num_src_par,
                                 int num_src, int *src_pos, double *src_flux,
                                 double *coldensh_out, double sig, double dr,
                                 const double *ndens, const double *xh_av,
                                 double *phi_ion, int m1,
                                 const double *photo_thin_table,
                                 const double *photo_thick_table, double minlogtau,
                                 double dlogtau, int num_tau, int last_l, int last_r);

}  // namespace asora
