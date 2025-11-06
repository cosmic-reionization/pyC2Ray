#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "tests.cuh"
#include "utils.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *asora_test_cinterp(PyObject *self, PyObject *args) {
    PyObject *pos0;
    PyArrayObject *dens;
    int i0, j0, k0;

    // Error checking
    if (!PyArg_ParseTuple(args, "OO", &pos0, &dens)) return NULL;
    if (!PyArg_ParseTuple(pos0, "iii", &i0, &j0, &k0)) return NULL;
    if (!PyArray_Check(dens) || PyArray_TYPE(dens) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "dens must be numpy array of type double");
        return NULL;
    }

    // Get density data
    const auto dens_data = static_cast<double *>(PyArray_DATA(dens));
    auto dens_size = static_cast<size_t>(PyArray_NBYTES(dens));
    auto m1 = static_cast<int>(PyArray_DIM(dens, 0));

    // Create output
    constexpr std::array<npy_intp, 4> out_shape = {4, 4, 4, 2};
    auto output = reinterpret_cast<PyArrayObject *>(
        PyArray_SimpleNew(out_shape.size(), out_shape.data(), NPY_DOUBLE)
    );
    auto out_data = static_cast<double *>(PyArray_DATA(output));

    // Run test kernel
    try {
        std::array<size_t, 4> cpp_shape;
        std::copy(out_shape.begin(), out_shape.end(), cpp_shape.begin());
        asoratest::cinterp_gpu(
            out_data, cpp_shape, dens_data, dens_size, {i0, j0, k0}, m1
        );
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return NULL;
    }

    return PyArray_Return(reinterpret_cast<PyArrayObject *>(output));
}

PyObject *asora_test_linthrd2cart(PyObject *self, PyObject *args) {
    int s, q;
    if (!PyArg_ParseTuple(args, "ii", &s, &q)) return NULL;

    auto [i, j, k] = asoratest::linthrd2cart(s, q);
    return Py_BuildValue("iii", i, j, k);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"cinterp", asora_test_cinterp, METH_VARARGS, "Geometric OCTA raytracing (GPU)"},
    {"linthrd2cart", asora_test_linthrd2cart, METH_VARARGS,
     "Shell mapping to cartesian coordinates"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasoratest",
    "Exposure of internal functions for testing purposes", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasoratest(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
