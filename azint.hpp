#pragma once 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

enum class Unit
{
    q,
    tth
};


struct Entry
{
    Entry(int c, float v) : col(c), value(v) {}
    int col;
    float value;
};

struct RListMatrix
{
    RListMatrix(int nrows) : rows(nrows), nelements(0) {}
    std::vector<std::vector<Entry> > rows;
    size_t nelements;
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

class Sparse
{
public:
    Sparse(py::object py_poni, 
           py::sequence py_shape, 
           float pixel_size,
           int n_splitting, 
           py::array_t<int8_t> mask,
           const std::string& unit,
           py::array_t<float, py::array::c_style | py::array::forcecast> radial_bins,
           std::optional<py::array_t<float, py::array::c_style | py::array::forcecast> > phi_bins);
    Sparse(std::vector<int>&& c,
           std::vector<int>&& r,
           std::vector<float>&& v,
           std::vector<float>&& vc,
           std::vector<float>&& vc2);
    void set_correction(py::array_t<float> corrections);
    py::array_t<float> spmv(py::array x);
    py::array_t<float> spmv_corrected(py::array x);
    py::array_t<float> spmv_corrected2(py::array x);
    // sparse csr matrix
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    std::vector<float> values;
    std::vector<float> values_corrected;
    std::vector<float> values_corrected2;
};
