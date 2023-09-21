#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "azint.hpp"

void tocsr(std::vector<RListMatrix>& segments, int nrows,
      std::vector<int>& col_idx, std::vector<int>& row_ptr, std::vector<float>& values)
{
    row_ptr.resize(nrows + 1);
    int nentry = 0;
    for (int i=0; i<nrows; i++) {
        row_ptr[i] = nentry;
        
        for (size_t rank=0; rank<segments.size(); rank++) {
            const std::vector<Entry>& row = segments[rank].rows[i];
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

void generate_matrix(long shape[2], int n_splitting, float pixel_size, 
                     std::vector<RListMatrix>& segments,
                     const Poni& poni, const int8_t* mask,
                     const Unit& output_unit,
                     int nradial_bins, const float* radial_bins,
                     int nphi_bins, float* phi_bins)
{
    float rot[3][3];
    rotation_matrix(rot, poni);
    
    #pragma omp parallel for schedule(static)
    for (int i=0; i<shape[0]; i++) {
        int rank = omp_get_thread_num();
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
                    
                    float radial_coord = 0.0f; 
                    switch(output_unit) {
                        case Unit::q:
                            // 4pi / lambda sin(2theta / 2) in A-1
                            radial_coord = 4.0e-10 * M_PI / poni.wavelength * sinf(0.5*tth);
                            break;

                        case Unit::tth:
                            radial_coord = tth;
                            break;
                    }

                    auto lower = std::lower_bound(radial_bins, 
                                                  radial_bins+nradial_bins+1, 
                                                  radial_coord);
                    int radial_index = std::distance(radial_bins, lower) - 1;
                    if ((radial_index < 0) || (radial_index >= nradial_bins)) {
                        continue;
                    }
                    int bin_index = radial_index;
                    
                    // 2D integration
                    if (phi_bins) {
                        // convert atan2 from [-pi, pi] to [0, 360] degrees
                        float phi = atan2f(-pos[0], -pos[1]) / M_PI*180.0f + 180.0f;
                        
                        auto lower = std::lower_bound(phi_bins, 
                                                      phi_bins+nphi_bins+1, 
                                                      phi);
                        int phi_index = std::distance(phi_bins, lower) - 1;
                        if ((phi_index < 0) || (phi_index >= nphi_bins)) {
                            continue;
                        }
                        bin_index += phi_index * nradial_bins;
                    }
                    
                    auto& row = segments[rank].rows[bin_index];
                    // sum duplicate entries
                    if (row.size() > 0 && (row.back().col == pixel_index)) {
                        row.back().value += 1.0f / (n_splitting * n_splitting);
                    }
                    else {
                        row.emplace_back(pixel_index, 1.0f / (n_splitting * n_splitting));
                    }
                }
            }
        }
    }
}

Sparse::Sparse(py::object py_poni, 
               py::sequence py_shape, 
               float pixel_size,
               int n_splitting, 
               py::array_t<int8_t> mask,
               const std::string& unit,
               py::array_t<float, py::array::c_style | py::array::forcecast> radial_bins,
               std::optional<py::array_t<float, py::array::c_style | py::array::forcecast> > phi_bins)
{
    Poni poni;
    poni.dist = py_poni.attr("dist").cast<float>();
    poni.poni1 = py_poni.attr("poni1").cast<float>();
    poni.poni2 = py_poni.attr("poni2").cast<float>();
    poni.rot1 = py_poni.attr("rot1").cast<float>();
    poni.rot2 = py_poni.attr("rot2").cast<float>();
    poni.rot3 = py_poni.attr("rot3").cast<float>();
    poni.wavelength = py_poni.attr("wavelength").cast<float>();
    
    long shape[] = {py_shape[0].cast<long>(), 
                    py_shape[1].cast<long>()};
                    
    Unit output_unit;
    if (unit == "q") {
        output_unit = Unit::q;
    }
    else {
        output_unit = Unit::tth;
    }
    
    int max_threads = omp_get_max_threads();
    std::vector<RListMatrix> segments;
    int nradial_bins = radial_bins.size() - 1;
    
    int nrows, nphi_bins;
    float* phi_data;
    // 1D integration
    if (!phi_bins.has_value()) {
        nrows = nradial_bins;
        nphi_bins = 0;
        phi_data = nullptr;
    }
    // 2D integration
    else {
        nphi_bins = phi_bins.value().size() - 1;
        nrows = nphi_bins * nradial_bins;
        phi_data = phi_bins.value().mutable_data();
    }
    
    segments.resize(max_threads, nrows);
    generate_matrix(shape, 
                    n_splitting, 
                    pixel_size, 
                    segments, 
                    poni, 
                    mask.data(),
                    output_unit,
                    nradial_bins, radial_bins.data(),
                    nphi_bins, phi_data);
    
    tocsr(segments, nrows, col_idx, row_ptr, values);
}

Sparse::Sparse(std::vector<int>&& c,
               std::vector<int>&& r,
               std::vector<float>&& v,
               std::vector<float>&& vc,
               std::vector<float>&& vc2) : col_idx(c), row_ptr(r), values(v), values_corrected(vc), values_corrected2(vc2)
{
}

void Sparse::set_correction(py::array_t<float> corrections)
{
    values_corrected.resize(values.size());
    values_corrected2.resize(values.size());
    const float* cdata = corrections.data();
    int nrows = row_ptr.size() - 1;
    for (long i=0; i<nrows; i++) {
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++) {
            // values_corrected = c / (solidangle * polarization)
            values_corrected[j] = values[j] / cdata[col_idx[j]];
            // values_corrected2 = c**2 / (solidangle**2 * polarization**2) for error progagation
            values_corrected2[j] = values[j] * values[j] / (cdata[col_idx[j]] * cdata[col_idx[j]]);
        }
    }
}


// sparse matrix vector multiplication A * x with sparse matrix A and vector x
template <typename T>
void _spmv(long nrows, const std::vector<int>& col_idx, 
           const std::vector<int>& row_ptr, 
           const std::vector<float>& values, 
           float* b, 
           const T* x)
{
    for (long i=0; i<nrows; i++) {
        double sum = 0.0;
        for (int j=row_ptr[i]; j<row_ptr[i+1]; j++) {
            sum += static_cast<double>(values[j]) * static_cast<double>(x[col_idx[j]]);
        }
        b[i] = static_cast<float>(sum);
    }
}

py::array_t<float> spmv(const std::vector<int>& col_idx, 
                        const std::vector<int>& row_ptr, 
                        const std::vector<float>& values,
                        const py::array& x)
{
    int nrows = row_ptr.size() - 1;
    py::array_t<float,  py::array::c_style> b(nrows);
    
    if (py::isinstance<py::array_t<uint8_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint8_t*)x.data());
    }
    else if (py::isinstance<py::array_t<uint16_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint16_t*)x.data());
    }
    else if (py::isinstance<py::array_t<uint32_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint32_t*)x.data());
    }
    else if (py::isinstance<py::array_t<int8_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int8_t*)x.data());
    }
    else if (py::isinstance<py::array_t<int16_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int16_t*)x.data());
    }
    else if (py::isinstance<py::array_t<int32_t>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int32_t*)x.data());
    }
    else if (py::isinstance<py::array_t<float>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (float*)x.data());
    }
    else if (py::isinstance<py::array_t<double>>(x)) {
        _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (double*)x.data());
    }
    return b;
}

py::array_t<float> Sparse::spmv(py::array x)
{
    return ::spmv(col_idx, row_ptr, values, x);
}

py::array_t<float> Sparse::spmv_corrected(py::array x)
{
    return ::spmv(col_idx, row_ptr, values_corrected, x);
}

py::array_t<float> Sparse::spmv_corrected2(py::array x)
{
    return ::spmv(col_idx, row_ptr, values_corrected2, x);
}

PYBIND11_MODULE(_azint, m) {
    py::class_<Sparse>(m, "Sparse")
        .def(py::init<py::object, py::sequence, float, int, py::array_t<int8_t>, std::string, py::array_t<float>, std::optional<py::array_t<float> > >())
        .def("set_correction", &Sparse::set_correction)
        .def("spmv", &Sparse::spmv)
        .def("spmv_corrected", &Sparse::spmv_corrected)
        .def("spmv_corrected2", &Sparse::spmv_corrected2)
        .def(py::pickle(
            [](const Sparse &s) {
                return py::make_tuple(s.col_idx, s.row_ptr, s.values, s.values_corrected, s.values_corrected2);
            },
            [](py::tuple t) {
                Sparse s(std::move(t[0].cast<std::vector<int> >()),
                         std::move(t[1].cast<std::vector<int> >()),
                         std::move(t[2].cast<std::vector<float> >()),
                         std::move(t[3].cast<std::vector<float> >()),
                         std::move(t[4].cast<std::vector<float> >()));
                return s;
            }
        ));
}
