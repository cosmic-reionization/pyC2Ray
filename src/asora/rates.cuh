#pragma once

namespace asora {

    // Photoionization rate from tables
    __device__ double photoion_rates_gpu(
        double strength, double coldens_in, double coldens_out, double Vfact,
        double sig, const double *thin_table, const double *thick_table,
        double minlogtau, double dlogtau, int num_tau
    );

    // Table interpolation lookup function
    __device__ double photo_lookuptable(
        const double *table, double tau, double minlogtau, double dlogtau, int num_tau
    );

    // Photoionization rates from analytical expression (grey-opacity)
    __device__ double photoion_rates_test_gpu(
        double strength, double coldens_in, double coldens_out, double Vfact, double sig
    );

}  // namespace asora
