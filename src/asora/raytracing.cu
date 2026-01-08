#include "raytracing.cuh"

#include "memory.h"
#include "rates.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <exception>

namespace {

    constexpr double max_coldensh = 2e30;

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    // Flat-array index from 3D (i,j,k) indices
    __device__ int mem_offset_gpu(int i, int j, int k, size_t N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

#if !defined(PERIODIC)
    // Check if point is in domain
    __device__ bool in_box_gpu(int i, int j, int k, int N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }
#endif

    __device__ void update_col_dens_phi_ion(
        double &coldens_out, double &phi_ion, double coldens_in, double nHI_p,
        double path, double strength, double vol, double sigma,
        const asora::photo_tables &ion_tables, const asora::linspace<double> &logtau
    ) {
        // Compute outgoing column density and add to array for subsequent
        // interpolations
        coldens_out = coldens_in + nHI_p * path;

#if defined(GREY_NOTABLES)
        auto phi = asora::photoion_rates_test_gpu(coldens_in, coldens_out, sigma);
#else
        auto phi = asora::photoion_rates_gpu(
            coldens_in, coldens_out, sigma, ion_tables, logtau
        );
#endif
        // Rescale the photo-ionization rate by the flux strength normalized per volume
        // and per neutral density (part of the photon-conserving rate prescription) and
        // add it to the global array
        atomicAdd(&phi_ion, phi * strength / vol / nHI_p);
    }

}  // namespace

namespace asora {

    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, double sigma, double dr, const double *xh_av, double *phi_ion,
        size_t num_src, size_t m1, double minlogtau, double dlogtau, size_t num_tau,
        size_t grid_size, size_t block_size
    ) {
        device::check_initialized();

        // Lazy allocation of memory that'll be used until the end of the application
        auto n_cells = m1 * m1 * m1;
        if (!device::contains(buffer_tag::photo_ionization))
            device::add<double>(buffer_tag::photo_ionization, n_cells);
        if (!device::contains(buffer_tag::hydrogen_fraction))
            device::add<double>(buffer_tag::hydrogen_fraction, n_cells);
        if (!device::contains(buffer_tag::column_density))
            device::add<double>(buffer_tag::column_density, grid_size * n_cells);

        //  Determine how large the octahedron should be, based on the raytracing
        //  radius. Currently, this is set s.t. the radius equals the distance from
        //  the source to the middle of the faces of the octahedron. To raytrace the
        //  whole box, the octahedron bust be 1.5*N in size
        int q_max = std::ceil(c::sqrt3<> * min(R, c::sqrt3<> * m1 / 2.0));

        // Here we fill the ionization rate array with zero before raytracing all
        // sources. The LOCALRATES flag is for debugging purposes and will be
        // removed later on
        auto phi_buf = device::get(buffer_tag::photo_ionization);
        auto phi_d = phi_buf.view<double>().data();
        safe_cuda(cudaMemset(phi_d, 0, phi_buf.size()));

        // density array is not modified, asora assumes that it has been copied to
        // the device before
        auto xh_buf = device::get(buffer_tag::hydrogen_fraction);
        auto xh_d = xh_buf.view<double>().data();
        xh_buf.copyFromHost(xh_av, xh_buf.size());

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        int last_r = m1 / 2 - 1 + modulo(m1, 2);
        int last_l = -m1 / 2;

        auto src_flux_d = device::get(buffer_tag::source_flux).view<double>().data();
        auto src_pos_d = device::get(buffer_tag::source_position).view<int>().data();
        auto cdh_d = device::get(buffer_tag::column_density).view<double>().data();
        auto n_d = device::get(buffer_tag::number_density).view<double>().data();
        auto ph_thin_d =
            device::get(buffer_tag::photo_thin_table).view<double>().data();
        auto ph_thick_d =
            device::get(buffer_tag::photo_thick_table).view<double>().data();

        // Loop over batches of sources
        for (size_t ns = 0; ns < num_src; ns += grid_size) {
            // Raytrace the current batch of sources in parallel
            // Consecutive kernel launches are in the same stream and so are serialized
            evolve0D_gpu<<<grid_size, block_size>>>(
                R, q_max, ns, num_src, src_pos_d, src_flux_d, cdh_d, sigma, dr, n_d,
                xh_d, phi_d, m1, ph_thin_d, ph_thick_d, minlogtau, dlogtau, num_tau,
                last_l, last_r
            );

            safe_cuda(cudaPeekAtLastError());
        }

        // Copy the accumulated ionization fraction back to the host
        // Memcpy blocks until last kernel has finished
        phi_buf.copyToHost(phi_ion, phi_buf.size());
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        double Rmax_LLS,
        int q_max,  // Is now the size of max q
        size_t ns_start, size_t num_src, int *__restrict__ src_pos,
        double *__restrict__ src_flux, double *__restrict__ coldensh_out, double sigma,
        double dr, const double *__restrict__ ndens, const double *__restrict__ xh_av,
        double *__restrict__ phi_ion, size_t m1,
        const double *__restrict__ photo_thin_table,
        const double *__restrict__ photo_thick_table, double minlogtau, double dlogtau,
        size_t num_tau, int last_l, int last_r
    ) {
        /* The raytracing kernel proceeds as follows:
            1. Select the source based on the block number (within the batch = the
           grid)
            2. Loop over the asora q-cells around the source, up to q_max (loop "A")
            3. Inside each shell, threads independently do all cells, possibly
           requiring multiple iterations if the block size is smaller than the
           number of cells in the shell (loop "B")
            4. After each shell, the threads are synchronized to ensure that
           causality is respected
        */

        // Source number = Start of batch + block number (each block does one
        // source)
        const int ns = ns_start + blockIdx.x;

        // Ensure the source index is valid
        if (ns >= num_src) return;

        asora::linspace<double> logtau{minlogtau, dlogtau, num_tau};
        asora::photo_tables ion_tables{photo_thin_table, photo_thick_table};

        // Get source properties
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block needs its own copy of the array)
        auto cdh_offset = blockIdx.x * m1 * m1 * m1;
        coldensh_out += cdh_offset;

        if (threadIdx.x == 0) {
            const auto offset = mem_offset_gpu(i0, j0, k0, m1);
            double nHI_p = ndens[offset] * (1.0 - xh_av[offset]);
            update_col_dens_phi_ion(
                coldensh_out[offset], phi_ion[offset], 0.0, nHI_p, 0.5 * dr, strength,
                dr * dr * dr, sigma, ion_tables, logtau
            );
        }
        __syncthreads();

        // (A) Loop over ASORA q-shells
        for (int q = 1; q <= q_max; q++) {
            // We figure out the number of cells in the shell and determine how many
            // passes the block needs to take to treat all of them
            int num_cells = 4 * q * q + 2;
            int passes = std::ceil(static_cast<float>(num_cells) / blockDim.x);

            // The threads have 1D indices 0,...,blocksize-1. We map these 1D
            // indices to the 3D positions of the cells inside the shell via the
            // mapping described in the paper. Since in general there are more cells
            // than threads, there is an additional loop here (B) so that all cells
            // are treated.
            for (int ipass = 0; ipass < passes; ipass++) {
                // "s" is the index in the 1D-range [0,...,4q^2 + 1] that gets
                // mapped to the cells in the shell
                int s = ipass * blockDim.x + threadIdx.x;

                // Ensure the thread maps to a valid cell
                if (s >= num_cells) continue;

                auto &&[i, j, k] = linthrd2cart(q, s);

                // Only do cell if it is within the (shifted under periodicity)
                // grid, i.e. at most ~N cells away from the source
                if ((i < last_l) || (i > last_r) || (j < last_l) || (j > last_r) ||
                    (k < last_l) || (k > last_r))
                    continue;

                auto dist2 =
                    (dr * i) * (dr * i) + (dr * j) * (dr * j) + (dr * k) * (dr * k);
                const bool at_origin = (i == 0) && (j == 0) && (k == 0);

                // Center to source
                i += i0;
                j += j0;
                k += k0;

#if !defined(PERIODIC)
                // When not in periodic mode, only treat cell if its in the grid
                if (!in_box_gpu(i, j, k, m1)) continue;
#endif
                auto [coldensh_in, path] =
                    cinterp_gpu(i, j, k, i0, j0, k0, coldensh_out, sigma, m1);
                path *= dr;
                auto vol_ph = 4 * c::pi<> * dist2 * path;

                // Get local ionization fraction & neutral hydrogen density in the cell
                const auto offset = mem_offset_gpu(i, j, k, m1);
                double nHI_p = ndens[offset] * (1.0 - xh_av[offset]);

                if (coldensh_in > max_coldensh) continue;

                // Reducing the following calculation changes the numerical precision of
                // the result, albeit the physical result doesn't.
                if (dist2 / (dr * dr) > Rmax_LLS * Rmax_LLS) continue;

                // Compute photoionization rates from column density.
                // WARNING: for now this is limited to the grey-opacity
                // test case source

                update_col_dens_phi_ion(
                    coldensh_out[offset], phi_ion[offset], coldensh_in, nHI_p, path,
                    strength, vol_ph, sigma, ion_tables, logtau
                );
            }
            __syncthreads();
        }
    }

    // dk is the largest delta.
    __device__ cuda::std::array<double, 5> geometric_factors(
        double di, double dj, double dk
    ) {
        auto path = sqrt(1.0 + (di * di + dj * dj) / (dk * dk));

        auto dx = abs(std::copysign(1.0, di) - di / std::abs(dk));
        auto dy = abs(std::copysign(1.0, dj) - dj / std::abs(dk));

        auto w1 = (1. - dx) * (1. - dy);
        auto w2 = (1. - dy) * dx;
        auto w3 = (1. - dx) * dy;
        auto w4 = dx * dy;

        return {path, w1, w2, w3, w4};
    }

    __device__ cuda::std::pair<double, double> cinterp_gpu(
        int i, int j, int k, int i0, int j0, int k0,
        const double *__restrict__ coldensh_out, double sigma_HI_at_ion_freq, size_t m1
    ) {
        auto di = i - i0;
        auto dj = j - j0;
        auto dk = k - k0;

        if (di == 0 && dj == 0 && dk == 0) return {0.0, 0.5};

        auto si = static_cast<int>(std::copysignf(1.f, di));
        auto sj = static_cast<int>(std::copysignf(1.f, dj));
        auto sk = static_cast<int>(std::copysignf(1.f, dk));
        auto ai = std::abs(di);
        auto aj = std::abs(dj);
        auto ak = std::abs(dk);

        auto get_column_density = [&coldensh_out, i, j, k,
                                   m1](int i_off, int j_off, int k_off) {
            return coldensh_out[mem_offset_gpu(i - i_off, j - j_off, k - k_off, m1)];
        };

        double c1, c2, c3, c4;
        if (ak >= aj && ak >= ai) {
            c1 = get_column_density(si, sj, sk);
            c2 = get_column_density(0, sj, sk);
            c3 = get_column_density(si, 0, sk);
            c4 = get_column_density(0, 0, sk);
        } else if (aj >= ai && aj >= ak) {
            c1 = get_column_density(si, sj, sk);
            c2 = get_column_density(0, sj, sk);
            c3 = get_column_density(si, sj, 0);
            c4 = get_column_density(0, sj, 0);
            cuda::std::swap(dj, dk);
            cuda::std::swap(aj, ak);
        } else {  // (ai >= aj && ai >= ak)
            c1 = get_column_density(si, sj, sk);
            c2 = get_column_density(si, 0, sk);
            c3 = get_column_density(si, sj, 0);
            c4 = get_column_density(si, 0, 0);
            cuda::std::swap(di, dk);
            cuda::std::swap(ai, ak);
            cuda::std::swap(di, dj);
            cuda::std::swap(ai, aj);
        }

        auto &&[path, w1, w2, w3, w4] = geometric_factors(
            static_cast<double>(di), static_cast<double>(dj), static_cast<double>(dk)
        );

        // Weight function for C2Ray interpolation function
        auto weightf_gpu = [sigma = sigma_HI_at_ion_freq](double cd) {
            constexpr double tau_0 = 0.6;
            return 1.0 / max(tau_0, cd * sigma);
        };

        w1 *= weightf_gpu(c1);
        w2 *= weightf_gpu(c2);
        w3 *= weightf_gpu(c3);
        w4 *= weightf_gpu(c4);

        // Column density at the crossing point
        auto cdensi = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);

        // Take care of diagonals
        if (ak == 1 && ai == 1 && aj == 1)
            cdensi *= c::sqrt3<>;
        else if (ak == 1 && (ai == 1 || aj == 1))
            cdensi *= c::sqrt2<>;

        return cuda::std::make_pair(cdensi, path);
    }

}  // namespace asora
