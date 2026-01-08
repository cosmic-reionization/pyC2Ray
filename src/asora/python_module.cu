#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "memory.h"
#include "raytracing.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

// ===========================================================================
// ASORA Python C-extension module
// Mostly boilerplate code, this file contains the wrappers for python
// to access the C++ functions of the ASORA library. Care has to be taken
// mostly with the numpy array arguments, since the underlying raw C pointer
// is passed directly to the C++ functions without additional type checking.
// ===========================================================================

// ========================================================================
// Raytrace all sources and compute photoionization rates
// ========================================================================
PyObject *asora_do_all_sources([[maybe_unused]] PyObject *self, PyObject *args) {
    double R;
    double sig;
    double dr;
    PyArrayObject *xh_av;
    PyArrayObject *phi_ion;
    size_t num_src;
    size_t m1;
    double minlogtau;
    double dlogtau;
    size_t num_tau;
    size_t grid_size;
    size_t block_size = 256;

    if (!PyArg_ParseTuple(
            args, "dddOOkkddkk|k", &R, &sig, &dr, &xh_av, &phi_ion, &num_src, &m1,
            &minlogtau, &dlogtau, &num_tau, &grid_size, &block_size
        ))
        return nullptr;

    // Error checking
    if (!PyArray_Check(xh_av) || PyArray_TYPE(xh_av) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "xh_av must be Array of type double");
        return nullptr;
    }
    if (!PyArray_Check(phi_ion) || PyArray_TYPE(phi_ion) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "phi_ion must be Array of type double");
        return nullptr;
    }

    // Get Array data
    auto xh_av_data = static_cast<double *>(PyArray_DATA(xh_av));
    auto phi_ion_data = static_cast<double *>(PyArray_DATA(phi_ion));

    try {
        asora::do_all_sources_gpu(
            R, sig, dr, xh_av_data, phi_ion_data, num_src, m1, minlogtau, dlogtau,
            num_tau, grid_size, block_size
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }

    return Py_None;
}

// Initialize GPU device and allocate some memory for grid data
PyObject *asora_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    unsigned int mpi_rank = 0;
    if (!PyArg_ParseTuple(args, "|I", &mpi_rank)) return nullptr;

    try {
        // Initialize the device
        asora::device::initialize(mpi_rank);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return Py_None;
}

// Close device and deallocate memory
PyObject *asora_device_close([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    try {
        asora::device::close();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
    return Py_None;
}

PyObject *asora_is_device_init([[maybe_unused]] PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) return nullptr;

    return asora::device::is_initialized() ? Py_True : Py_False;
}

namespace {

    template <typename T>
    NPY_TYPES getNpyType();

    template <>
    NPY_TYPES getNpyType<double>() {
        return NPY_DOUBLE;
    }

    template <>
    NPY_TYPES getNpyType<int>() {
        return NPY_INT;
    }

    template <typename T>
    bool load_array_to_device(const PyArrayObject *array, asora::buffer_tag tag) {
        if (!PyArray_Check(array) || PyArray_TYPE(array) != getNpyType<T>()) {
            using namespace std::string_literals;
            std::string msg =
                "array must be a numpy NDArray of type "s + typeid(T).name();
            PyErr_SetString(PyExc_TypeError, msg.c_str());
            return false;
        }

        auto data = static_cast<T *>(PyArray_DATA(array));
        auto size = static_cast<size_t>(PyArray_SIZE(array));

        try {
            asora::device::transfer<T>(tag, data, size);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_TypeError, e.what());
            return false;
        }
        return true;
    }

}  // namespace

// Copy density grid to GPU
PyObject *asora_density_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *ndens;
    return PyArg_ParseTuple(args, "O", &ndens) &&  //
                   load_array_to_device<double>(
                       ndens, asora::buffer_tag::number_density
                   )
               ? Py_None
               : nullptr;
}

// Copy radiation tables to GPU
PyObject *asora_photo_table_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *thin_table, *thick_table;
    return PyArg_ParseTuple(args, "OO", &thin_table, &thick_table) &&
                   load_array_to_device<double>(
                       thin_table, asora::buffer_tag::photo_thin_table
                   ) &&
                   load_array_to_device<double>(
                       thick_table, asora::buffer_tag::photo_thick_table
                   )
               ? Py_None
               : nullptr;
}

// Copy source data to GPU
PyObject *asora_source_data_to_device([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *src_pos, *src_flux;
    return PyArg_ParseTuple(args, "OO", &src_pos, &src_flux) &&
                   load_array_to_device<int>(
                       src_pos, asora::buffer_tag::source_position
                   ) &&
                   load_array_to_device<double>(
                       src_flux, asora::buffer_tag::source_flux
                   )
               ? Py_None
               : nullptr;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"do_all_sources", asora_do_all_sources, METH_VARARGS, "Do OCTA raytracing (GPU)"},
    {"device_init", asora_device_init, METH_VARARGS,
     "Initialize device and allocate memory"},
    {"device_close", asora_device_close, METH_VARARGS, "Close device and free memory"},
    {"is_device_init", asora_is_device_init, METH_VARARGS,
     "Check if the device is initialized"},
    {"density_to_device", asora_density_to_device, METH_VARARGS,
     "Copy density field to GPU"},
    {"photo_table_to_device", asora_photo_table_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {"source_data_to_device", asora_source_data_to_device, METH_VARARGS,
     "Copy radiation table to GPU"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasora",
    "CUDA C++ implementation of the short-characteristics RT", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasora(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
