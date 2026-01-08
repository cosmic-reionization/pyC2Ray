#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "tests.cuh"
#include "utils.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *asora_test_cinterp([[maybe_unused]] PyObject *self, PyObject *args) {
    PyObject *pos0;
    PyArrayObject *dens;
    int i0, j0, k0;

    // Error checking
    if (!PyArg_ParseTuple(args, "OO", &pos0, &dens)) return nullptr;
    if (!PyArg_ParseTuple(pos0, "iii", &i0, &j0, &k0)) return nullptr;
    if (!PyArray_Check(dens) || PyArray_TYPE(dens) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "dens must be numpy array of type double");
        return nullptr;
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
        return nullptr;
    }

    return PyArray_Return(reinterpret_cast<PyArrayObject *>(output));
}

PyObject *asora_test_linthrd2cart([[maybe_unused]] PyObject *self, PyObject *args) {
    int q, s;
    if (!PyArg_ParseTuple(args, "ii", &q, &s)) return nullptr;

    auto [i, j, k] = asoratest::linthrd2cart(q, s);
    return Py_BuildValue("iii", i, j, k);
}

PyObject *asora_test_cart2linthrd([[maybe_unused]] PyObject *self, PyObject *args) {
    int i, j, k;
    if (!PyArg_ParseTuple(args, "iii", &i, &j, &k)) return nullptr;

    auto [q, s] = asoratest::cart2linthrd(i, j, k);
    return Py_BuildValue("ii", q, s);
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
     "Shell indexing to cartesian coordinates"},
    {"cart2linthrd", asora_test_cart2linthrd, METH_VARARGS,
     "Cartesian coordinates to shell indexing"},
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
