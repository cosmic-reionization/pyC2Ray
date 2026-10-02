#include "raytracing.cuh"

#include "memory.cuh"
#include "rates.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <exception>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define FOURPI 12.566370614359172463991853874177  // 4π
#define INV4PI 0.079577471545947672804111050482   // 1/4π
#define SQRT3 1.73205080757                       // Square root of 3
#define SQRT2 1.41421356237                       // Square root of 2
#define MAX_COLDENSH 2e30    // Column density limit (rates are set to zero above this)
#define CUDA_BLOCK_SIZE 256  // Size of blocks used to treat sources

// ========================================================================
// Utility Device Functions
// ========================================================================

namespace {

    // Fortran-type modulo function (C modulo is signed)
    __host__ __device__ int modulo(int a, int b) { return (a % b + b) % b; }

    // Sign function on the device
    __host__ __device__ int sign(double x) { return x >= 0 ? 1 : -1; }

    // Flat-array index from 3D (i,j,k) indices
    __device__ int mem_offst_gpu(int i, int j, int k, int N) {
        return N * N * modulo(i, N) + N * modulo(j, N) + modulo(k, N);
    }

    // Mapping from cartesian coordinates of a cell to reduced cache memory space
    // (here N = 2qmax + 1 in general)
    [[maybe_unused]] __device__ int cart2cache(int i, int j, int k, int N) {
        return N * N * int(k < 0) + N * i + j;
    }

    // Mapping from linear 1D indices to the cartesian coords of a q-shell in asora
    __device__ void linthrd2cart(int s, int q, int &i, int &j) {
        if (s == 0) {
            i = q;
            j = 0;
            return;
        }

        int b = (s - 1) / (2 * q);
        int a = (s - 1) % (2 * q);

        if (a + 2 * b > 2 * q) {
            a = a + 1;
            b = b - 1 - q;
        }

        i = a + b - q;
        j = b;
    }

// When using a GPU with compute capability < 6.0, we must manually define the
// atomicAdd function for doubles
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    static __inline__ __device__ double atomicAdd(double *address, double val) {
        unsigned long long int *address_as_ull = (unsigned long long int *)address;
        unsigned long long int old = *address_as_ull, assumed;
        if (val == 0.0) return __longlong_as_double(old);
        do {
            assumed = old;
            old = atomicCAS(
                address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed))
            );
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif

    // Check if point is in domain (deprecated)
    [[maybe_unused]] __device__ bool in_box_gpu(int i, int j, int k, int N) {
        return (i >= 0 && i < N) && (j >= 0 && j < N) && (k >= 0 && k < N);
    }

}  // namespace

namespace asora {

    // ========================================================================
    // Raytrace all sources and add up ionization rates
    // ========================================================================
    void do_all_sources_gpu(
        double R, double *coldensh_out, double sig, double dr, double *ndens,
        double *xh_av, double *phi_ion, int num_src, int m1, double minlogtau,
        double dlogtau, int num_tau
    ) {
        // Byte-size of grid data
        auto meshsize = m1 * m1 * m1 * sizeof(double);

        //  Determine how large the octahedron should be, based on the raytracing
        //  radius. Currently, this is set s.t. the radius equals the distance from
        //  the source to the middle of the faces of the octahedron. To raytrace the
        //  whole box, the octahedron bust be 1.5*N in size
        int max_q = std::ceil(SQRT3 * min(R, SQRT3 * m1 / 2.0));

        // CUDA Grid size: since 1 block = 1 source, this sets the number of sources
        // treated in parallel
        dim3 gs(NUM_SRC_PAR);

        // CUDA Block size: more of a tuning parameter (see above), in practice
        // anything ~128 is fine
        dim3 bs(CUDA_BLOCK_SIZE);

        try {
            // Here we fill the ionization rate array with zero before raytracing all
            // sources. The LOCALRATES flag is for debugging purposes and will be
            // removed later on
            safe_cuda(cudaMemset(phi_dev, 0, meshsize));

            // Copy current ionization fraction to the device
            // cudaMemcpy(n_dev,ndens,meshsize,cudaMemcpyHostToDevice);  < --- !!
            // density array is not modified, asora assumes that it has been copied to
            // the device before
            safe_cuda(cudaMemcpy(x_dev, xh_av, meshsize, cudaMemcpyHostToDevice));
        } catch (const std::exception &) {
            return;
        }

        // Since the grid is periodic, we limit the maximum size of the raytraced
        // region to a cube as large as the mesh around the source. See line 93 of
        // evolve_source in C2Ray, this size will depend on if the mesh is even or
        // odd. Basically the idea is that you never touch a cell which is outside a
        // cube of length ~N centered on the source
        int last_r = m1 / 2 - 1 + modulo(m1, 2);
        int last_l = -m1 / 2;

        // Loop over batches of sources
        for (int ns = 0; ns < num_src; ns += NUM_SRC_PAR) {
            // Raytrace the current batch of sources in parallel
            evolve0D_gpu<<<gs, bs>>>(
                R, max_q, ns, num_src, NUM_SRC_PAR, src_pos_dev, src_flux_dev, cdh_dev,
                sig, dr, n_dev, x_dev, phi_dev, m1, photo_thin_table_dev,
                photo_thick_table_dev, minlogtau, dlogtau, num_tau, last_l, last_r
            );

            try {
                safe_cuda(cudaPeekAtLastError());
                // Sync device to be sure (is this required ??)
                safe_cuda(cudaDeviceSynchronize());
            } catch (const std::exception &) {
                return;
            }
        }

        try {
            // Copy the accumulated ionization fraction back to the host
            safe_cuda(cudaMemcpy(phi_ion, phi_dev, meshsize, cudaMemcpyDeviceToHost));
            safe_cuda(
                cudaMemcpy(coldensh_out, cdh_dev, meshsize, cudaMemcpyDeviceToHost)
            );
        } catch (const std::exception &) {
        }
    }

    // ========================================================================
    // Raytracing kernel, adapted from C2Ray. Calculates in/out column density
    // to the current cell and finds the photoionization rate
    // ========================================================================
    __global__ void evolve0D_gpu(
        double Rmax_LLS,
        int q_max,  // Is now the size of max q
        int ns_start, int num_src, int num_src_par, int *src_pos, double *src_flux,
        double *coldensh_out, double sig, double dr, const double *ndens,
        const double *xh_av, double *phi_ion, int m1, const double *photo_thin_table,
        const double *photo_thick_table, double minlogtau, double dlogtau, int num_tau,
        int last_l, int last_r
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

        // Get source properties
        const auto i0 = src_pos[3 * ns + 0];
        const auto j0 = src_pos[3 * ns + 1];
        const auto k0 = src_pos[3 * ns + 2];
        const auto strength = src_flux[ns];

        // Offset pointer to the outgoing column density array used for
        // interpolation (each block needs its own copy of the array)
        auto cdh_offset = blockIdx.x * m1 * m1 * m1;
        coldensh_out += cdh_offset;

        // (A) Loop over ASORA q-shells
        for (int q = 0; q <= q_max; q++) {
            // We figure out the number of cells in the shell and determine how many
            // passes the block needs to take to treat all of them
            int num_cells = 4 * q * q + 2;
            int Npass = num_cells / blockDim.x + 1;

            // The threads have 1D indices 0,...,blocksize-1. We map these 1D
            // indices to the 3D positions of the cells inside the shell via the
            // mapping described in the paper. Since in general there are more cells
            // than threads, there is an additional loop here (B) so that all cells
            // are treated.
            int s_end = q > 0 ? num_cells : 1;
            int s_end_top = 2 * q * (q + 1) + 1;

            // (B) Loop over cells in the shell
            for (int ipass = 0; ipass < Npass; ipass++) {
                // "s" is the index in the 1D-range [0,...,4q^2 + 1] that gets
                // mapped to the cells in the shell
                int s = ipass * blockDim.x + threadIdx.x;

                // Ensure the thread maps to a valid cell
                if (s >= s_end) continue;

                int i, j, k;
                int sgn;
                // Determine if cell is in top or bottom part of the shell (the
                // mapping is slightly different due to the part that is on the
                // same z-plane as the source)
                if (s < s_end_top) {
                    sgn = 1;
                    linthrd2cart(s, q, i, j);
                } else {
                    sgn = -1;
                    linthrd2cart(s - s_end_top, q - 1, i, j);
                }
                k = sgn * q - sgn * (abs(i) + abs(j));

                // Only do cell if it is within the (shifted under periodicity)
                // grid, i.e. at most ~N cells away from the source
                if ((i < last_l) || (i > last_r) || (j < last_l) || (j > last_r) ||
                    (k < last_l) || (k > last_r))
                    continue;

                // TODO: early exit on distance can be done here:
                // auto dist2 = i * i + j * j + k * k;
                // if (dist2 > Rmax_LLS * Rmax_LLS) continue;

                // Center to source
                i += i0;
                j += j0;
                k += k0;

// When not in periodic mode, only treat cell if its in the grid
#if !defined(PERIODIC)
                if (!in_box_gpu(i, j, k, m1)) continue;
#endif
                // Map to periodic grid
                auto offset = mem_offst_gpu(i, j, k, m1);

                // Get local ionization fraction & neutral Hydrogen density in the cell
                double xh_av_p = xh_av[offset];
                double nHI_p = ndens[offset] * (1.0 - xh_av_p);

                // PH (29.9.23): There used to be a check here if the
                // coldensh_out of the current cell was zero to
                // "determine if it hasn't been done before". I think
                // this isn't necessary anymore in this version and
                // eliminates the need to set the array to zero between
                // source batches, which for large batches is a
                // SIGNIFICANT bottleneck.

                double coldensh_in;  // Column density to the cell
                double path;
                double vol_ph;
                double dist2;

                // If its the source cell, just find path (no incoming
                // column density)
                if (i == i0 && j == j0 && k == k0) {
                    coldensh_in = 0.0;
                    path = 0.5 * dr;
                    vol_ph = dr * dr * dr;
                    dist2 = 0.0;
                }
                // If its another cell, do interpolation to find
                // incoming column density
                else {
                    cuda::std::tie(coldensh_in, path) =
                        cinterp_gpu(i, j, k, i0, j0, k0, coldensh_out, sig, m1);
                    path *= dr;
                    auto xs = dr * (i - i0);
                    auto ys = dr * (j - j0);
                    auto zs = dr * (k - k0);
                    dist2 = xs * xs + ys * ys + zs * zs;
                    vol_ph = dist2 * path * FOURPI;
                }

                // Compute outgoing column density and add to array for
                // subsequent interpolations
                double cdho = coldensh_in + nHI_p * path;
                coldensh_out[offset] = cdho;

                // Compute photoionization rates from column density.
                // WARNING: for now this is limited to the grey-opacity
                // test case source
                if (coldensh_in > MAX_COLDENSH) continue;
                if (dist2 / (dr * dr) > Rmax_LLS * Rmax_LLS) continue;

#if defined(GREY_NOTABLES)
                auto phi = photoion_rates_test_gpu(
                    strength, coldensh_in, coldensh_out[offset], vol_ph, sig
                );
#else
                auto phi = photoion_rates_gpu(
                    strength, coldensh_in, cdho, vol_ph, sig, photo_thin_table,
                    photo_thick_table, minlogtau, dlogtau, num_tau
                );
#endif
                // Divide the photo-ionization rates by the
                // appropriate neutral density (part of the
                // photon-conserving rate prescription)
                phi /= nHI_p;

                // Add the computed ionization rate and the column
                // density to the array ATOMICALLY since multiple
                // blocks could be writing to the same cell at the
                // same time!
                atomicAdd(&phi_ion[offset], phi);
                // atomicAdd(coldensh_out +
                // mem_offst_gpu(pos[0],pos[1],pos[2],m1),cdho);
                // TODO: this seems to induce some double precision
                // floating error
            }
            // IMPORTANT: Sync threads after each shell so that the next only begins
            // when all outgoing column densities of the current shell are available
            __syncthreads();
        }
    }

    // dk is the largest delta.
    __device__ cuda::std::array<double, 5> geometric_factors(
        double di, double dj, double dk
    ) {
        auto path = sqrt(1.0 + (di * di + dj * dj) / (dk * dk));

        auto dx = abs(sign(di) - di / abs(dk));
        auto dy = abs(sign(dj) - dj / abs(dk));

        auto s1 = (1. - dx) * (1. - dy);
        auto s2 = (1. - dy) * dx;
        auto s3 = (1. - dx) * dy;
        auto s4 = dx * dy;

        return {path, s1, s2, s3, s4};
    }

    __device__ cuda::std::pair<double, double> cinterp_gpu(
        int i, int j, int k, int i0, int j0, int k0, const double *coldensh_out,
        double sigma_HI_at_ion_freq, int m1
    ) {
        auto di = i - i0;
        auto dj = j - j0;
        auto dk = k - k0;

        auto ai = abs(di);
        auto aj = abs(dj);
        auto ak = abs(dk);
        auto si = sign(di);
        auto sj = sign(dj);
        auto sk = sign(dk);

        auto get_column_density = [&coldensh_out, i, j, k,
                                   m1](int i_off, int j_off, int k_off) {
            return coldensh_out[mem_offst_gpu(i - i_off, j - j_off, k - k_off, m1)];
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

        auto &&[path, s1, s2, s3, s4] = geometric_factors(
            static_cast<double>(di), static_cast<double>(dj), static_cast<double>(dk)
        );

        // Weight function for C2Ray interpolation function
        auto weightf_gpu = [](double cd, double sig) {
            constexpr double tau_0 = 0.6;
            return 1.0 / max(tau_0, cd * sig);
        };

        auto w1 = s1 * weightf_gpu(c1, sigma_HI_at_ion_freq);
        auto w2 = s2 * weightf_gpu(c2, sigma_HI_at_ion_freq);
        auto w3 = s3 * weightf_gpu(c3, sigma_HI_at_ion_freq);
        auto w4 = s4 * weightf_gpu(c4, sigma_HI_at_ion_freq);

        // Column density at the crossing point
        auto cdensi = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);

        // Take care of diagonals
        if (ak == 1 && ai == 1 && aj == 1)
            cdensi *= SQRT3;
        else if (ak == 1 && (ai == 1 || aj == 1))
            cdensi *= SQRT2;

        return cuda::std::make_pair(cdensi, path);
    }

}  // namespace asora
