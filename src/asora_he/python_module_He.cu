#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "memory_He.cuh"
#include "raytracing_He.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

// ===========================================================================
// ASORA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the ASORA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

extern "C" {
// ========================================================================
// Raytrace all sources and compute photoionization rates
// ========================================================================
static PyObject *asora_do_all_sources(PyObject *self, PyObject *args) {
    double R;
    PyArrayObject *coldensh_out;
    PyArrayObject *coldenshei_out;
    PyArrayObject *coldensheii_out;
    PyArrayObject *sig_hi;
    PyArrayObject *sig_hei;
    PyArrayObject *sig_heii;
    int nbin1;
    int nbin2;
    int nbin3;
    double dr;
    PyArrayObject *ndens;
    PyArrayObject *xHI_av;
    PyArrayObject *xHeI_av;
    PyArrayObject *xHeII_av;
    PyArrayObject *phi_ion_HI;
    PyArrayObject *phi_ion_HeI;
    PyArrayObject *phi_ion_HeII;
    PyArrayObject *phi_heat_HI;
    PyArrayObject *phi_heat_HeI;
    PyArrayObject *phi_heat_HeII;
    int NumSrc;
    int m1;
    double minlogtau;
    double dlogtau;
    int NumTau;

    if (!PyArg_ParseTuple(
            args, "dOOOOOOiiidOOOOOOOOOOiiddi", &R, &coldensh_out, &coldenshei_out,
            &coldensheii_out, &sig_hi, &sig_hei, &sig_heii, &nbin1, &nbin2, &nbin3, &dr,
            &ndens, &xHI_av, &xHeI_av, &xHeII_av, &phi_ion_HI, &phi_ion_HeI,
            &phi_ion_HeII, &phi_heat_HI, &phi_heat_HeI, &phi_heat_HeII, &NumSrc, &m1,
            &minlogtau, &dlogtau, &NumTau
        ))
        return NULL;

    // Error checking
    if (!PyArray_Check(coldensh_out) || PyArray_TYPE(coldensh_out) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "coldensh_out must be Array of type double");
        return NULL;
    } else if (!PyArray_Check(coldenshei_out) ||
               PyArray_TYPE(coldenshei_out) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "coldenshei_out must be Array of type double");
        return NULL;
    } else if (!PyArray_Check(coldensheii_out) ||
               PyArray_TYPE(coldensheii_out) != NPY_DOUBLE) {
        PyErr_SetString(
            PyExc_TypeError, "coldensheii_out must be Array of type double"
        );
        return NULL;
    }

    // Get Array data
    double *coldensh_out_hi = (double *)PyArray_DATA(coldensh_out);
    double *coldensh_out_hei = (double *)PyArray_DATA(coldenshei_out);
    double *coldensh_out_heii = (double *)PyArray_DATA(coldensheii_out);
    double *ndens_data = (double *)PyArray_DATA(ndens);
    double *phi_ion_HI_data = (double *)PyArray_DATA(phi_ion_HI);
    double *phi_ion_HeI_data = (double *)PyArray_DATA(phi_ion_HeI);
    double *phi_ion_HeII_data = (double *)PyArray_DATA(phi_ion_HeII);
    double *phi_heat_HI_data = (double *)PyArray_DATA(phi_heat_HI);
    double *phi_heat_HeI_data = (double *)PyArray_DATA(phi_heat_HeI);
    double *phi_heat_HeII_data = (double *)PyArray_DATA(phi_heat_HeII);
    double *xh_av_HI_data = (double *)PyArray_DATA(xHI_av);
    double *xh_av_HeI_data = (double *)PyArray_DATA(xHeI_av);
    double *xh_av_HeII_data = (double *)PyArray_DATA(xHeII_av);
    double *sig_hi_data = (double *)PyArray_DATA(sig_hi);
    double *sig_hei_data = (double *)PyArray_DATA(sig_hei);
    double *sig_heii_data = (double *)PyArray_DATA(sig_heii);

    do_all_sources_gpu(
        R, coldensh_out_hi, coldensh_out_hei, coldensh_out_heii, sig_hi_data,
        sig_hei_data, sig_heii_data, nbin1, nbin2, nbin3, dr, ndens_data, xh_av_HI_data,
        xh_av_HeI_data, xh_av_HeII_data, phi_ion_HI_data, phi_ion_HeI_data,
        phi_ion_HeII_data, phi_heat_HI_data, phi_heat_HeI_data, phi_heat_HeII_data,
        NumSrc, m1, minlogtau, dlogtau, NumTau
    );

    return Py_None;
}

// ========================================================================
// Allocate GPU memory for grid data
// ========================================================================
static PyObject *asora_device_init(PyObject *self, PyObject *args) {
    int N;
    int num_src_par;
    int num_freq;
    if (!PyArg_ParseTuple(args, "iii", &N, &num_src_par, &num_freq)) return NULL;
    device_init(N, num_src_par, num_freq);
    return Py_None;
}

// ========================================================================
// Deallocate GPU memory
// ========================================================================
static PyObject *asora_device_close(PyObject *self, PyObject *args) {
    device_close();
    return Py_None;
}

// ========================================================================
// Copy density grid to GPU
// ========================================================================
static PyObject *asora_density_to_device(PyObject *self, PyObject *args) {
    int N;
    PyArrayObject *ndens;
    if (!PyArg_ParseTuple(args, "Oi", &ndens, &N)) return NULL;

    double *ndens_data = (double *)PyArray_DATA(ndens);
    density_to_device(ndens_data, N);

    return Py_None;
}

// ========================================================================
// Copy radiation table to GPU
// ========================================================================
static PyObject *asora_tables_to_device(PyObject *self, PyObject *args) {
    int NumTau;
    int NumFreq;
    PyArrayObject *photo_thin_table;
    PyArrayObject *photo_thick_table;
    PyArrayObject *heat_thin_table;
    PyArrayObject *heat_thick_table;
    if (!PyArg_ParseTuple(
            args, "OOOOii", &photo_thin_table, &photo_thick_table, &heat_thin_table,
            &heat_thick_table, &NumTau, &NumFreq
        ))
        return NULL;

    double *photo_thin_table_data = (double *)PyArray_DATA(photo_thin_table);
    double *photo_thick_table_data = (double *)PyArray_DATA(photo_thick_table);
    double *heat_thin_table_data = (double *)PyArray_DATA(heat_thin_table);
    double *heat_thick_table_data = (double *)PyArray_DATA(heat_thick_table);
    tables_to_device(
        photo_thin_table_data, photo_thick_table_data, heat_thin_table_data,
        heat_thick_table_data, NumTau, NumFreq
    );

    return Py_None;
}

// ========================================================================
// Copy source data to GPU
// ========================================================================
static PyObject *asora_source_data_to_device(PyObject *self, PyObject *args) {
    int NumSrc;
    PyArrayObject *pos;
    PyArrayObject *flux;
    if (!PyArg_ParseTuple(args, "OOi", &pos, &flux, &NumSrc)) return NULL;

    int *pos_data = (int *)PyArray_DATA(pos);
    double *flux_data = (double *)PyArray_DATA(flux);

    source_data_to_device(pos_data, flux_data, NumSrc);

    return Py_None;
}

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Do OCTA raytracing (GPU)"},
    {"device_init", asora_device_init, METH_VARARGS, "Free GPU memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Free GPU memory"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to GPU"},
    {"tables_to_device", asora_tables_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora_He",                      /* name of module */
    "CUDA C++ implementation of the short-characteristics RT", /* module
                                                                  documentation,
                                                                  may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    asoraMethods
};

PyMODINIT_FUNC PyInit_libasora_He(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}
}
