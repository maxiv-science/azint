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
    Entry(size_t c, double v) : col(c), value(v) {}
    size_t col;
    double value;
};

struct RListMatrix
{
    RListMatrix(int nrows) : rows(nrows), nelements(0) {}
    std::vector<std::vector<Entry> > rows;
    size_t nelements;
};

struct Poni
{
    double dist;
    double poni1;
    double poni2;
    double rot1;
    double rot2;
    double rot3;
    double wavelength;
};

class Sparse
{
public:
    Sparse(py::object py_poni,
           py::array_t<double> pixel_corners,
           const size_t n_splitting, 
           py::array_t<int8_t> mask,
           const std::string& unit,
           py::array_t<double, py::array::c_style | py::array::forcecast> radial_bins,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast> > phi_bins);
    Sparse(std::vector<size_t>&& c,
           std::vector<size_t>&& r,
           std::vector<double>&& v,
           std::vector<double>&& vc,
           std::vector<double>&& vc2);
    void set_correction(py::array_t<double> corrections);
    py::array_t<double> spmv(py::array x);
    py::array_t<double> spmv_corrected(py::array x);
    py::array_t<double> spmv_corrected2(py::array x);

  std::tuple<py::array_t<double>,py::array_t<double>>
  spmv_correctedPair(py::array x);


  // sparse csr matrix
    std::vector<size_t> col_idx;
    std::vector<size_t> row_ptr;
    std::vector<double> values;
    std::vector<double> values_corrected;
    std::vector<double> values_corrected2;
};
