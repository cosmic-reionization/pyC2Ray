#pragma once
#include <cuda_runtime.h>

// Photoionization rate from tables
__device__ double photoion_rates_gpu(
    const double &strength, const double &tau_in, const double &tau_out, const int &nf,
    const double &Vfact, const double *thin_table, const double *thick_table,
    const double &minlogtau, const double &dlogtau, const int &NumTau,
    const int &NumFreq
);

// Photoheating rate from tables
__device__ double photoheat_rates_gpu(
    const double &strength, const double &tau_in, const double &tau_out, const int &nf,
    const double &Vfact, const double *thin_table, const double *thick_table,
    const double &minlogtau, const double &dlogtau, const int &NumTau,
    const int &NumFreq
);

// Table interpolation lookup function
__device__ double photo_lookuptable(
    const double *, const int &, const double &, const double &, const double &,
    const int &
);

// Photoionization rates from analytical expression (grey-opacity)
__device__ double photoion_rates_test_gpu(
    const double &strength, const double &coldens_in, const double &coldens_out,
    const double &Vfact, const double &sig
);