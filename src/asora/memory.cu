#include "memory.cuh"

#include "utils.cuh"

#include <cuda_runtime.h>

#include <format>
#include <iostream>

namespace asora {

    // ========================================================================
    // Global variables. Pointers to GPU memory to store grid data
    //
    // To avoid uneccessary memory movement between host and device, we
    // allocate dedicated memory on the device via a call to device_init at the
    // beginning of the program. Data is copied to and from the host memory
    // (typically numpy arrays) only when it changes and is required. For example:
    //
    // * The density field is copied to the device only when it
    // actually changes, i.e. at the beginning of a timestep.
    // * The photoionization rates for each source are computed and summed
    // directly on the device and are copied to the host only when all sources
    // have been passed.
    // * The column density is NEVER copied back to the host, since it is only
    // accessed on the device when computing ionization rates.
    // ========================================================================
    double *cdh_dev;                // Outgoing column density of the cells
    double *n_dev;                  // Density
    double *x_dev;                  // Time-averaged ionized fraction
    double *phi_dev;                // Photoionization rates
    double *photo_thin_table_dev;   // Thin Radiation table
    double *photo_thick_table_dev;  // Thick Radiation table
    int *src_pos_dev;
    double *src_flux_dev;

    int NUM_SRC_PAR;

    // ========================================================================
    // Initialization function to allocate device memory (pointers above)
    // ========================================================================
    void device_init(int N, int num_src_par, int mpi_rank, int num_gpus) {
        // int dev_id = 0;

        // Here num_gpus is the number of gpus per node
        auto dev_id = mpi_rank % num_gpus;
        std::cout << "Number of GPUS " << num_gpus << "; selected ID " << dev_id
                  << "\n";

        // Explicitly set the device before querying
        cudaDeviceProp device_prop;
        try {
            safe_cuda(cudaSetDevice(dev_id));
            safe_cuda(cudaGetDeviceProperties(&device_prop, dev_id));
        } catch (const std::exception &) {
            return;
        }

        auto device_info =
            std::format("GPU Device ID {}: {} with compute  capability {}.{}", dev_id,
                        device_prop.name, device_prop.major, device_prop.minor);
        if (num_gpus > 1) std::cout << "MPI Rank " << mpi_rank << " has ";
        std::cout << device_info << "\n";

        // Byte-size of grid data
        auto bytesize = N * N * N * sizeof(double);

        // Set the source batch size, i.e. the number of sources done in parallel
        // (on the same GPU)
        NUM_SRC_PAR = num_src_par;

        // Allocate memory
        try {
            safe_cuda(cudaMalloc(&cdh_dev, NUM_SRC_PAR * bytesize));
            safe_cuda(cudaMalloc(&n_dev, bytesize));
            safe_cuda(cudaMalloc(&x_dev, bytesize));
            safe_cuda(cudaMalloc(&phi_dev, bytesize));
        } catch (const std::exception &) {
            return;
        }

        std::cout << "Successfully allocated " << (3 + NUM_SRC_PAR) * bytesize / 1e6
                  << " Mb of device memory for grid of size N = " << N
                  << ", with source batch size " << NUM_SRC_PAR << "\n";
    }

    // ========================================================================
    // Utility functions to copy data to device
    // ========================================================================
    void density_to_device(double *ndens, int N) {
        try {
            safe_cuda(cudaMemcpy(n_dev, ndens, N * N * N * sizeof(double),
                                 cudaMemcpyHostToDevice));
        } catch (const std::exception &) {
        }
    }

    void photo_table_to_device(double *thin_table, double *thick_table, int num_tau) {
        auto bytesize = num_tau * sizeof(double);
        try {
            // Copy thin table
            safe_cuda(cudaMalloc(&photo_thin_table_dev, bytesize));
            safe_cuda(cudaMemcpy(photo_thin_table_dev, thin_table, bytesize,
                                 cudaMemcpyHostToDevice));
            // Copy thick table
            safe_cuda(cudaMalloc(&photo_thick_table_dev, bytesize));
            safe_cuda(cudaMemcpy(photo_thick_table_dev, thick_table, bytesize,
                                 cudaMemcpyHostToDevice));
        } catch (const std::exception &) {
        }
    }

    void source_data_to_device(int *pos, double *flux, int num_src) {
        // Free arrays from previous evolve call
        try {
            // FIXME: not required?
            safe_cuda(cudaFree(src_pos_dev));
            safe_cuda(cudaFree(src_flux_dev));

            // Copy positions
            safe_cuda(cudaMalloc(&src_pos_dev, 3 * num_src * sizeof(int)));
            safe_cuda(cudaMemcpy(src_pos_dev, pos, 3 * num_src * sizeof(int),
                                 cudaMemcpyHostToDevice));

            // Copy strengths
            safe_cuda(cudaMalloc(&src_flux_dev, num_src * sizeof(double)));
            safe_cuda(cudaMemcpy(src_flux_dev, flux, num_src * sizeof(double),
                                 cudaMemcpyHostToDevice));
        } catch (const std::exception &) {
        }
    }

    // ========================================================================
    // Deallocate device memory at the end of a run
    // ========================================================================
    void device_close() {
        std::cout << "Deallocating device memory...\n";
        try {
            safe_cuda(cudaFree(cdh_dev));
            safe_cuda(cudaFree(n_dev));
            safe_cuda(cudaFree(x_dev));
            safe_cuda(cudaFree(phi_dev));
            safe_cuda(cudaFree(photo_thin_table_dev));
            safe_cuda(cudaFree(src_pos_dev));
            safe_cuda(cudaFree(src_flux_dev));
        } catch (const std::exception &) {
        }
    }

}  // namespace asora
