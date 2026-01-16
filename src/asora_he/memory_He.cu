#include "memory_He.cuh"

#include <iostream>

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
double *cdhei_dev;              // Outgoing column density of the cells
double *cdheii_dev;             // Outgoing column density of the cells
double *n_dev;                  // Density
double *xHI_dev;                // Time-averaged ionized fraction
double *xHeI_dev;               // Time-averaged ionized fraction
double *xHeII_dev;              // Time-averaged ionized fraction
double *phi_HI_dev;             // Photoionization rates
double *phi_HeI_dev;            // Photoionization rates
double *phi_HeII_dev;           // Photoionization rates
double *heat_HI_dev;            // Photoheating rates
double *heat_HeI_dev;           // Photoheating rates
double *heat_HeII_dev;          // Photoheating rates
double *photo_thin_table_dev;   // Thin Radiation table for photo-ionization
double *photo_thick_table_dev;  // Thick Radiation table for photo-ionization
double *heat_thin_table_dev;    // Thin Radiation table for photo-heating
double *heat_thick_table_dev;   // Thick Radiation table for photo-heating
int *src_pos_dev;
double *src_flux_dev;
double *sig_hi_dev;    // cross section at different frequencies
double *sig_hei_dev;   // cross section at different frequencies
double *sig_heii_dev;  // cross section at different frequencies

int NUM_SRC_PAR;
int NUM_FREQ;

// ========================================================================
// Initialization function to allocate device memory (pointers above)
// ========================================================================
void device_init(const int &N, const int &num_src_par, const int &num_freq) {
    int dev_id = 0;

    cudaDeviceProp device_prop;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&device_prop, dev_id);
    if (device_prop.computeMode == cudaComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                     "threads can use ::cudaSetDevice()"
                  << std::endl;
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties returned error code " << error
                  << ", line(" << __LINE__ << ")" << std::endl;
    } else {
        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
                  << "\" with compute capability " << device_prop.major << "."
                  << device_prop.minor << std::endl;
    }

    // Byte-size of grid and frequency data
    long unsigned int bytsize_grid = N * N * N * sizeof(double);
    long unsigned int bytsize_freq = num_freq * sizeof(double);

    // Set the source batch size, i.e. the number of sources done in parallel (on
    // the same GPU)
    NUM_SRC_PAR = num_src_par;
    NUM_FREQ = num_freq;

    // Allocate memory
    cudaMalloc(&cdh_dev, NUM_SRC_PAR * bytsize_grid);
    cudaMalloc(&cdhei_dev, NUM_SRC_PAR * bytsize_grid);
    cudaMalloc(&cdheii_dev, NUM_SRC_PAR * bytsize_grid);
    cudaMalloc(&n_dev, bytsize_grid);
    cudaMalloc(&xHI_dev, bytsize_grid);
    cudaMalloc(&xHeI_dev, bytsize_grid);
    cudaMalloc(&xHeII_dev, bytsize_grid);
    cudaMalloc(&phi_HI_dev, bytsize_grid);
    cudaMalloc(&phi_HeI_dev, bytsize_grid);
    cudaMalloc(&phi_HeII_dev, bytsize_grid);
    cudaMalloc(&heat_HI_dev, bytsize_grid);
    cudaMalloc(&heat_HeI_dev, bytsize_grid);
    cudaMalloc(&heat_HeII_dev, bytsize_grid);
    cudaMalloc(&sig_hi_dev, bytsize_freq);
    cudaMalloc(&sig_hei_dev, bytsize_freq);
    cudaMalloc(&sig_heii_dev, bytsize_freq);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            "Couldn't allocate memory: " +
            std::to_string(
                (3 * bytsize_freq + (7 + 3 * NUM_SRC_PAR) * bytsize_grid) / 1e6
            ) +
            std::string(cudaGetErrorName(error)) + " - " +
            std::string(cudaGetErrorString(error))
        );
    } else {
        // TODO: add message that tells also how many frequencies ...
        std::cout << "Succesfully allocated "
                  << (3 * bytsize_freq + (7 + 3 * NUM_SRC_PAR) * bytsize_grid) / 1e6
                  << " Mb of device memory for grid of size N = " << N;
        std::cout << ", with source batch size " << NUM_SRC_PAR << " and " << NUM_FREQ
                  << " frequency bins." << std::endl;
    }
}

// ========================================================================
// Utility functions to copy data to device
// ========================================================================
void density_to_device(double *ndens, const int &N) {
    cudaMemcpy(n_dev, ndens, N * N * N * sizeof(double), cudaMemcpyHostToDevice);
}

void tables_to_device(
    double *photo_thin_table, double *photo_thick_table, double *heat_thin_table,
    double *heat_thick_table, const int &NumTau, const int &NumFreq
) {
    // Copy thin table
    cudaMalloc(&photo_thin_table_dev, int(NumTau * NumFreq) * sizeof(double));
    cudaMalloc(&heat_thin_table_dev, int(NumTau * NumFreq) * sizeof(double));
    cudaMemcpy(
        photo_thin_table_dev, photo_thin_table, int(NumTau * NumFreq) * sizeof(double),
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        heat_thin_table_dev, heat_thin_table, int(NumTau * NumFreq) * sizeof(double),
        cudaMemcpyHostToDevice
    );

    // Copy thick table
    cudaMalloc(&photo_thick_table_dev, int(NumTau * NumFreq) * sizeof(double));
    cudaMalloc(&heat_thick_table_dev, int(NumTau * NumFreq) * sizeof(double));
    cudaMemcpy(
        photo_thick_table_dev, photo_thick_table,
        int(NumTau * NumFreq) * sizeof(double), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        heat_thick_table_dev, photo_thick_table, int(NumTau * NumFreq) * sizeof(double),
        cudaMemcpyHostToDevice
    );
}

void source_data_to_device(int *pos, double *flux, const int &NumSrc) {
    // Free arrays from previous evolve call
    cudaFree(src_pos_dev);
    cudaFree(src_flux_dev);

    // Allocate memory for sources of current evolve call
    cudaMalloc(&src_pos_dev, 3 * NumSrc * sizeof(int));
    cudaMalloc(&src_flux_dev, NumSrc * sizeof(double));

    // Copy source data (positions & strengths) to device
    cudaMemcpy(src_pos_dev, pos, 3 * NumSrc * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(src_flux_dev, flux, NumSrc * sizeof(double), cudaMemcpyHostToDevice);
}

// ========================================================================
// Deallocate device memory at the end of a run
// ========================================================================
void device_close() {
    printf("Deallocating device memory...\n");
    cudaFree(cdh_dev);
    cudaFree(cdhei_dev);
    cudaFree(cdheii_dev);
    cudaFree(n_dev);
    cudaFree(xHI_dev);
    cudaFree(xHeI_dev);
    cudaFree(xHeII_dev);
    cudaFree(phi_HI_dev);
    cudaFree(phi_HeI_dev);
    cudaFree(phi_HeII_dev);
    cudaFree(heat_HI_dev);
    cudaFree(heat_HeI_dev);
    cudaFree(heat_HeII_dev);
    cudaFree(photo_thick_table_dev);
    cudaFree(photo_thin_table_dev);
    cudaFree(heat_thick_table_dev);
    cudaFree(heat_thin_table_dev);
    cudaFree(src_pos_dev);
    cudaFree(src_flux_dev);
}
