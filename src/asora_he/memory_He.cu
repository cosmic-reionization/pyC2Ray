#include "memory_He.cuh"

#include "hip/hip_runtime.h"

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

    hipDeviceProp_t device_prop;
    hipGetDevice(&dev_id);
    hipGetDeviceProperties(&device_prop, dev_id);
    if (device_prop.computeMode == hipComputeModeProhibited) {
        std::cerr << "Error: device is running in <Compute Mode Prohibited>, no "
                     "threads can use ::hipSetDevice()"
                  << std::endl;
    }

    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        std::cout << "hipGetDeviceProperties returned error code " << error << ", line(" << __LINE__
                  << ")" << std::endl;
    } else {
        std::cout << "GPU Device " << dev_id << ": \"" << device_prop.name
                  << "\" with compute capability " << device_prop.major << "." << device_prop.minor
                  << std::endl;
    }

    // Byte-size of grid and frequency data
    long unsigned int bytsize_grid = N * N * N * sizeof(double);
    long unsigned int bytsize_freq = num_freq * sizeof(double);

    // Set the source batch size, i.e. the number of sources done in parallel (on
    // the same GPU)
    NUM_SRC_PAR = num_src_par;
    NUM_FREQ = num_freq;

    // Allocate memory
    hipMalloc(&cdh_dev, NUM_SRC_PAR * bytsize_grid);
    hipMalloc(&cdhei_dev, NUM_SRC_PAR * bytsize_grid);
    hipMalloc(&cdheii_dev, NUM_SRC_PAR * bytsize_grid);
    hipMalloc(&n_dev, bytsize_grid);
    hipMalloc(&xHI_dev, bytsize_grid);
    hipMalloc(&xHeI_dev, bytsize_grid);
    hipMalloc(&xHeII_dev, bytsize_grid);
    hipMalloc(&phi_HI_dev, bytsize_grid);
    hipMalloc(&phi_HeI_dev, bytsize_grid);
    hipMalloc(&phi_HeII_dev, bytsize_grid);
    hipMalloc(&heat_HI_dev, bytsize_grid);
    hipMalloc(&heat_HeI_dev, bytsize_grid);
    hipMalloc(&heat_HeII_dev, bytsize_grid);
    hipMalloc(&sig_hi_dev, bytsize_freq);
    hipMalloc(&sig_hei_dev, bytsize_freq);
    hipMalloc(&sig_heii_dev, bytsize_freq);

    error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(
            "Couldn't allocate memory: " +
            std::to_string((3 * bytsize_freq + (7 + 3 * NUM_SRC_PAR) * bytsize_grid) / 1e6) +
            std::string(hipGetErrorName(error)) + " - " + std::string(hipGetErrorString(error)));
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
    hipMemcpy(n_dev, ndens, N * N * N * sizeof(double), hipMemcpyHostToDevice);
}

void tables_to_device(double *photo_thin_table, double *photo_thick_table, double *heat_thin_table,
                      double *heat_thick_table, const int &NumTau, const int &NumFreq) {
    // Copy thin table
    hipMalloc(&photo_thin_table_dev, int(NumTau * NumFreq) * sizeof(double));
    hipMalloc(&heat_thin_table_dev, int(NumTau * NumFreq) * sizeof(double));
    hipMemcpy(photo_thin_table_dev, photo_thin_table, int(NumTau * NumFreq) * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(heat_thin_table_dev, heat_thin_table, int(NumTau * NumFreq) * sizeof(double),
              hipMemcpyHostToDevice);

    // Copy thick table
    hipMalloc(&photo_thick_table_dev, int(NumTau * NumFreq) * sizeof(double));
    hipMalloc(&heat_thick_table_dev, int(NumTau * NumFreq) * sizeof(double));
    hipMemcpy(photo_thick_table_dev, photo_thick_table, int(NumTau * NumFreq) * sizeof(double),
              hipMemcpyHostToDevice);
    hipMemcpy(heat_thick_table_dev, photo_thick_table, int(NumTau * NumFreq) * sizeof(double),
              hipMemcpyHostToDevice);
}

void source_data_to_device(int *pos, double *flux, const int &NumSrc) {
    // Free arrays from previous evolve call
    hipFree(src_pos_dev);
    hipFree(src_flux_dev);

    // Allocate memory for sources of current evolve call
    hipMalloc(&src_pos_dev, 3 * NumSrc * sizeof(int));
    hipMalloc(&src_flux_dev, NumSrc * sizeof(double));

    // Copy source data (positions & strengths) to device
    hipMemcpy(src_pos_dev, pos, 3 * NumSrc * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(src_flux_dev, flux, NumSrc * sizeof(double), hipMemcpyHostToDevice);
}

// ========================================================================
// Deallocate device memory at the end of a run
// ========================================================================
void device_close() {
    printf("Deallocating device memory...\n");
    hipFree(cdh_dev);
    hipFree(cdhei_dev);
    hipFree(cdheii_dev);
    hipFree(n_dev);
    hipFree(xHI_dev);
    hipFree(xHeI_dev);
    hipFree(xHeII_dev);
    hipFree(phi_HI_dev);
    hipFree(phi_HeI_dev);
    hipFree(phi_HeII_dev);
    hipFree(heat_HI_dev);
    hipFree(heat_HeI_dev);
    hipFree(heat_HeII_dev);
    hipFree(photo_thick_table_dev);
    hipFree(photo_thin_table_dev);
    hipFree(heat_thick_table_dev);
    hipFree(heat_thin_table_dev);
    hipFree(src_pos_dev);
    hipFree(src_flux_dev);
}
