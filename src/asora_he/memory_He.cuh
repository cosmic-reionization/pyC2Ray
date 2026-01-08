#pragma once

// Allocate grid memory
void device_init(const int &, const int &, const int &);

// Deallocate grid memory
void device_close();

// Copy density grid to device memory
void density_to_device(double *, const int &);

// Copy radiation tables to device memory
void tables_to_device(double *, double *, double *, double *, const int &, const int &);

// Copy source positions & fluxes to device memory
void source_data_to_device(int *, double *, const int &);

// Copy cross section array to device memory
void crossect_to_device(double *, const int &);

// Pointers to device memory
extern double *cdh_dev;
extern double *cdhei_dev;
extern double *cdheii_dev;
extern double *n_dev;
extern double *xHI_dev;
extern double *xHeI_dev;
extern double *xHeII_dev;
extern double *phi_HI_dev;
extern double *phi_HeI_dev;
extern double *phi_HeII_dev;
extern double *heat_HI_dev;
extern double *heat_HeI_dev;
extern double *heat_HeII_dev;
extern double *photo_thin_table_dev;
extern double *photo_thick_table_dev;
extern double *heat_thin_table_dev;
extern double *heat_thick_table_dev;
extern int *src_pos_dev;
extern double *src_flux_dev;
extern double *sig_hi_dev;
extern double *sig_hei_dev;
extern double *sig_heii_dev;

// Number of sources done in parallel ("source batch size")
extern int NUM_SRC_PAR;
extern int NUM_FREQ;