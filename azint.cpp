#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PY_ARRAY_UNIQUE_SYMBOL azint_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <structmember.h>
#include "vector.h"

struct Entry
{
    int col;
    float value;
};

struct Poni
{
    float dist;
    float poni1;
    float poni2;
    float rot1;
    float rot2;
    float rot3;
    float wavelength;
};

int bisect_right(int nbins, float* bins, float x)
{
    int lo = 0;
    int hi = nbins;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (x < bins[mid]) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }
    return lo;
}

double get_double(PyObject* obj, const char* key)
{
    PyObject* tmp = PyObject_GetAttrString(obj, key);
    if (tmp == NULL) {
        printf("tmp in get_double is null\n");
    }
    double value = PyFloat_AsDouble(tmp);
    Py_DECREF(tmp);
    return value;
}

void tocsr(Vector<Entry>* rows, int nrows,
      Vector<int>& col_idx, Vector<int>& row_ptr, Vector<float>& values)
{
    row_ptr.resize(nrows + 1);
    int nentry = 0;
    for (int i=0; i<nrows; i++) {
        row_ptr[i] = nentry;
        
        Vector<Entry>& row = rows[i];
        size_t j=0;
        while (j < row.size()) {
            int col = row[j].col;
            float value = row[j].value;
            //printf("value %.2f\n", value);
            j++;
            // sum duplicate entries
            while (j < row.size() && row[j].col == col) {
                value += row[j].value;
                j++;
            }
            col_idx.push_back(col);
            values.push_back(value);
            nentry++;
        }
    }
    row_ptr[nrows] = nentry;
}

// b = A*x
void dot(float b[3], float A[3][3], float x[3])
{
    b[0] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
    b[1] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
    b[2] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];
}

// A = B*C
void matrix_multiplication(float A[3][3], float B[3][3], float C[3][3])
{
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            float sum = 0.0;
            for (int k=0; k<3; k++) {
                sum += B[i][k] * C[k][j];
            }
            A[i][j] = sum;
        }
    }
}

void rotation_matrix(float rot[3][3], Poni poni)
{
    //Rotation about axis 1: Note this rotation is left-handed
    float rot1[3][3] = {{1.0, 0.0, 0.0},
                        {0.0, cosf(poni.rot1), sinf(poni.rot1)},
                        {0.0, -sinf(poni.rot1), cosf(poni.rot1)}};
                        
    // Rotation about axis 2. Note this rotation is left-handed
    float rot2[3][3] = {{cosf(poni.rot2), 0.0, -sinf(poni.rot2)},
                        {0.0, 1.0, 0.0},
                        {sinf(poni.rot2), 0.0, cosf(poni.rot2)}};
                        
    // Rotation about axis 3: Note this rotation is right-handed
    float rot3[3][3] = {{cosf(poni.rot3), -sinf(poni.rot3), 0.0},
                        {sinf(poni.rot3), cosf(poni.rot3), 0.0},
                        {0.0, 0.0, 1.0}};
    float tmp[3][3];
    // np.dot(np.dot(rot3, rot2), rot1)
    matrix_multiplication(tmp, rot3, rot2);
    matrix_multiplication(rot, tmp, rot1);
}

int get_shape(PyObject* py_shape, long shape[2])
{
    PyObject* seq = PySequence_Fast(py_shape, "expected a sequence for shape");
    if (seq == NULL) {
        return 1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != 2) {
        PyErr_SetString(PyExc_TypeError, "shape must have two dimensions");
        return 1;
    }
    shape[0] = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, 0));
    shape[1] = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, 1));
    Py_DECREF(seq);
    return 0;
}

void generate_matrix_1d(long shape[2], int n_splitting, float pixel_size, Vector<Entry>* rows,
                      Poni& poni, int8_t* mask, int nradial_bins, float* radial_bins)
{
    float rot[3][3];
    rotation_matrix(rot, poni);
    
    for (int i=0; i<shape[0]; i++) {
        for (int j=0; j<shape[1]; j++) {
            int pixel_index = i*shape[1] + j;
            if (mask[pixel_index]) {
                continue;
            }
            for (int k=0; k<n_splitting; k++) {
                for (int l=0; l<n_splitting; l++) {
                    float p[] = {
                        (i + (k + 0.5f) / n_splitting) * pixel_size - poni.poni1,
                        (j + (l + 0.5f) / n_splitting) * pixel_size - poni.poni2,
                        poni.dist
                    };
                    float pos[3];
                    dot(pos, rot, p);
                    
                    float r = sqrtf(pos[0]*pos[0] + pos[1]*pos[1]);
                    float tth = atan2f(r, pos[2]);
                    // q = 4pi/lambda sin( 2theta / 2 ) in nm-1
                    float q = 4.0e-9 * M_PI / poni.wavelength * sinf(0.5*tth);
                    int radial_index = bisect_right(nradial_bins, radial_bins, q) - 1;
                    if ((radial_index < 0) || (radial_index >= nradial_bins)) {
                        continue;
                    }
                    
                    int bin_index = radial_index;
                    Entry entry = {pixel_index, 1.0f / (n_splitting * n_splitting)};
                    rows[bin_index].push_back(entry);
                }
            }
        }
    }
}

void generate_matrix_2d(long shape[2], int n_splitting, float pixel_size, Vector<Entry>* rows,
                      Poni& poni, int8_t* mask, 
                      int nradial_bins, float* radial_bins,
                      int nphi_bins, float* phi_bins)
{
    float rot[3][3];
    rotation_matrix(rot, poni);
    
    for (int i=0; i<shape[0]; i++) {
        for (int j=0; j<shape[1]; j++) {
            int pixel_index = i*shape[1] + j;
            if (mask[pixel_index]) {
                continue;
            }
            for (int k=0; k<n_splitting; k++) {
                for (int l=0; l<n_splitting; l++) {
                    float p[] = {
                        (i + (k + 0.5f) / n_splitting) * pixel_size - poni.poni1,
                        (j + (l + 0.5f) / n_splitting) * pixel_size - poni.poni2,
                        poni.dist
                    };
                    float pos[3];
                    dot(pos, rot, p);
                    
                    float r = sqrtf(pos[0]*pos[0] + pos[1]*pos[1]);
                    float tth = atan2f(r, pos[2]);
                    // q = 4pi/lambda sin( 2theta / 2 ) in nm-1
                    float q = 4.0e-9 * M_PI / poni.wavelength * sinf(0.5*tth);
                    int radial_index = bisect_right(nradial_bins, radial_bins, q) - 1;
                    if ((radial_index < 0) || (radial_index >= nradial_bins)) {
                        continue;
                    }
                    
                    float phi = atan2f(pos[0], pos[1]);
                    int phi_index = bisect_right(nphi_bins, phi_bins, phi) - 1;
                    if ((phi_index < 0) || (phi_index >= nphi_bins)) {
                        continue;
                    }
                    int bin_index = phi_index * nradial_bins + radial_index;
                    Entry entry = {pixel_index, 1.0f / (n_splitting * n_splitting)};
                    rows[bin_index].push_back(entry);
                }
            }
        }
    }
}

static PyObject*
generate_matrix(PyObject* self, PyObject* args)
{
    PyObject* py_poni;
    PyObject* py_shape;
    int n_splitting;
    float pixel_size;
    PyArrayObject* py_mask;
    PyObject* py_bins;
    if (!PyArg_ParseTuple(args, "OOfiO!O", 
        &py_poni, 
        &py_shape,
        &pixel_size,
        &n_splitting, 
        &PyArray_Type, &py_mask, 
        &py_bins)) {
        return NULL;
    }
    
    long shape[2];
    if (get_shape(py_shape, shape)) {
        return NULL;
    }
    
    PyArrayObject* mask_array = (PyArrayObject*)PyArray_FromAny(
        (PyObject*)py_mask, 
        PyArray_DescrFromType(NPY_INT8), 
        1, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST, NULL);
    
    if (mask_array == NULL) {
        printf("Error in PyArray_FromAny\n");
    }
    int8_t* mask = (int8_t*)PyArray_DATA(mask_array);
    
    Poni poni;
    poni.dist = get_double(py_poni, "dist");
    poni.poni1 = get_double(py_poni, "poni1");
    poni.poni2 = get_double(py_poni, "poni2");
    poni.rot1 = get_double(py_poni, "rot1");
    poni.rot2 = get_double(py_poni, "rot2");
    poni.rot3 = get_double(py_poni, "rot3");
    poni.wavelength = get_double(py_poni, "wavelength");
    
    int nrows;
    Vector<Entry>* rows;
    
    PyObject* seq = PySequence_Fast(py_bins, "expected a sequence for bins");
    if (seq == NULL) {
        return NULL;
    }
    
    PyArrayObject* radial_array = (PyArrayObject*)PyArray_FromAny(
        PySequence_Fast_GET_ITEM(seq, 0), 
        PyArray_DescrFromType(NPY_FLOAT32), 
        1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST, NULL);
    int nradial_bins = PyArray_SIZE(radial_array) - 1;
    float* radial_bins = (float*)PyArray_DATA(radial_array);
    
    // 1D integration
    if (PySequence_Fast_GET_SIZE(seq) == 1) {
        nrows = nradial_bins;
        rows = new Vector<Entry>[nrows];
        generate_matrix_1d(shape, n_splitting, pixel_size, rows, 
                           poni, mask, nradial_bins, radial_bins);
    }
    // 2D integration
    else if (PySequence_Fast_GET_SIZE(seq) == 2) {
        PyArrayObject* phi_array = (PyArrayObject*)PyArray_FromAny(
            PySequence_Fast_GET_ITEM(seq, 1), 
            PyArray_DescrFromType(NPY_FLOAT32), 
            1, 1, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_FORCECAST, NULL);
        
        int nphi_bins = PyArray_SIZE(phi_array) - 1;
        float* phi_bins = (float*)PyArray_DATA(phi_array);
        nrows = nphi_bins * nradial_bins;
        rows = new Vector<Entry>[nrows];
        generate_matrix_2d(shape, n_splitting, pixel_size, rows, poni, mask, 
                           nradial_bins, radial_bins,
                           nphi_bins, phi_bins);
        Py_DECREF(phi_array);
    }
    else {
        PyErr_SetString(PyExc_ValueError, "bins is tuple of radial and optionally phi bins");
        return NULL;
    }
        
    Py_DECREF(seq);
    Py_DECREF(radial_array);
    Py_DECREF(mask_array);
    
    Vector<int> col_idx, row_ptr;
    Vector<float> values;
    col_idx.leak();
    row_ptr.leak();
    values.leak();
    tocsr(rows, nrows, col_idx, row_ptr, values);
    delete [] rows;
    
    PyObject* tuple = PyTuple_New(3);
    
    npy_intp dims[] = {static_cast<long>(col_idx.size())};
    PyObject* py_array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, col_idx.data());
    PyArray_ENABLEFLAGS((PyArrayObject*) py_array, NPY_ARRAY_OWNDATA);
    PyTuple_SetItem(tuple, 0, py_array);
    
    dims[0] = row_ptr.size();
    py_array = PyArray_SimpleNewFromData(1, dims, NPY_INT32, row_ptr.data());
    PyArray_ENABLEFLAGS((PyArrayObject*) py_array, NPY_ARRAY_OWNDATA);
    PyTuple_SetItem(tuple, 1, py_array);
    
    dims[0] = values.size();
    py_array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, values.data());
    PyArray_ENABLEFLAGS((PyArrayObject*) py_array, NPY_ARRAY_OWNDATA);
    PyTuple_SetItem(tuple, 2, py_array);
    return tuple;
}

template <typename T>
void _spmv(long nrows, int* col_idx, int* row_ptr, float* values, float* b, T* x)
{
    for (long i=0; i<nrows; i++) {
        b[i] = 0.0f;
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++) {
            b[i] += values[j] * x[col_idx[j]];
        }
    }
}

static PyObject*
spmv(PyObject* self, PyObject* args)
{
    PyArrayObject* col_idx_array;
    PyArrayObject* row_ptr_array;
    PyArrayObject* values_array;
    PyArrayObject* x_array;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
        &PyArray_Type, &col_idx_array, 
        &PyArray_Type, &row_ptr_array, 
        &PyArray_Type, &values_array,
        &PyArray_Type, &x_array)) {
        printf("Error parsing\n");
        return NULL;
    }
    
    npy_intp nrows = PyArray_SIZE(row_ptr_array) - 1;
    npy_intp shape[] = {nrows};
    PyObject* py_array = PyArray_SimpleNew(1, shape, NPY_FLOAT32);
    
    int* col_idx = (int*)PyArray_DATA(col_idx_array);
    int* row_ptr = (int*)PyArray_DATA(row_ptr_array);
    float* values = (float*)PyArray_DATA(values_array);
    float* b = (float*)PyArray_DATA((PyArrayObject*)py_array);
    
    int type = PyArray_TYPE(x_array);
    switch(type) {
        case NPY_UINT16: {
            uint16_t* x = (uint16_t*)PyArray_DATA(x_array);
            _spmv(nrows, col_idx, row_ptr, values, b, x);
            break;
        }
        case NPY_INT16: {
            int16_t* x = (int16_t*)PyArray_DATA(x_array);
            _spmv(nrows, col_idx, row_ptr, values, b, x);
            break;
        }
        case NPY_UINT32: {
            uint32_t* x = (uint32_t*)PyArray_DATA(x_array);
            _spmv(nrows, col_idx, row_ptr, values, b, x);
            break;
        }
        case NPY_INT32: {
            int32_t* x = (int32_t*)PyArray_DATA(x_array);
            _spmv(nrows, col_idx, row_ptr, values, b, x);
            break;
        }
        case NPY_FLOAT32: {
            float* x = (float*)PyArray_DATA(x_array);
            _spmv(nrows, col_idx, row_ptr, values, b, x);
            break;
        }
        default: {
            PyErr_SetString(PyExc_ValueError, "Wrong type");
            return NULL;
        }
    }
    return py_array;
}

static PyMethodDef AzintMethods[] = {
    {"generate_matrix", generate_matrix, METH_VARARGS,
     ""},
     {"spmv", spmv, METH_VARARGS,
     "dot product"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef azintmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_azint",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
    .m_methods = AzintMethods
};

PyMODINIT_FUNC
PyInit__azint(void)
{
    import_array();
    return PyModule_Create(&azintmodule);
}
