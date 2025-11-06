#pragma once

#include <array>

namespace asoratest {

    void cinterp_gpu(
        double *out_data, const std::array<size_t, 4> &out_shape,
        const double *dens_data, size_t dens_size, const int3 &pos0, int m1
    );

    std::array<int, 3> linthrd2cart(int s, int q);

};  // namespace asoratest
