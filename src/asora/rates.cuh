#pragma once

namespace asora {

    template <typename T>
    struct linspace {
        T start;
        T step;
        size_t num;

        __host__ __device__ T stop() const { return start + num * step; }
    };

    struct photo_tables {
        const double *thin;
        const double *thick;
    };

    // Photoionization rate from tables
    __device__ double photoion_rates_gpu(
        double coldens_in, double coldens_out, double sigma,
        const photo_tables &ion_tables, const linspace<double> &logtau
    );

#ifdef GREY_NOTABLES
    // Photoionization rates from analytical expression (grey-opacity)
    __device__ double photoion_rates_test_gpu(
        double coldens_in, double coldens_out, double sigma
    );
#endif  // GREY_NOTABLES

}  // namespace asora
