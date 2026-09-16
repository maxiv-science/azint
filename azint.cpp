#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
// #include "Timer.h"
#include "azint.hpp"

void tocsr(const std::vector<RListMatrix>& segments,
	   const size_t nrows,
	   std::vector<size_t>& col_idx,
	   std::vector<size_t>& row_ptr,
	   std::vector<double>& values)
{
    row_ptr.resize(nrows+1);
    size_t nentry = 0;
    for (size_t i=0; i<nrows; i++)
      {
        row_ptr[i] = nentry;

	for(const RListMatrix& rankSegment : segments)
	  {
            const std::vector<Entry>& row =
	      rankSegment.rows[i];

            size_t j=0;
            while (j<row.size())
	      {
                const size_t col = row[j].col;
                double value = row[j].value;
                j++;
                // sum duplicate entries
                while (j<row.size() &&
		       row[j].col == col)
		  {
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
void dot(double b[3], double A[3][3], double x[3])
{
    b[0] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
    b[1] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
    b[2] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];
}

// A = B*C
void matrix_multiplication(double A[3][3], double B[3][3], double C[3][3])
{
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            double sum = 0.0;
            for (int k=0; k<3; k++) {
                sum += B[i][k] * C[k][j];
            }
            A[i][j] = sum;
        }
    }
}

void rotation_matrix(double rot[3][3],const Poni& poni)
{
    //Rotation about axis 1: Note this rotation is left-handed
  const double rot1[3][3] = {{1.0, 0.0, 0.0},
			 {0.0, std::cos(poni.rot1), std::sin(poni.rot1)},
			 {0.0, -std::sin(poni.rot1), std::cos(poni.rot1)}};
                        
    // Rotation about axis 2. Note this rotation is left-handed
  const double rot2[3][3] = {{std::cos(poni.rot2), 0.0, -std::sin(poni.rot2)},
                        {0.0, 1.0, 0.0},
                        {std::sin(poni.rot2), 0.0, std::cos(poni.rot2)}};
                        
    // Rotation about axis 3: Note this rotation is right-handed
  const double rot3[3][3] = {{std::cos(poni.rot3), -std::sin(poni.rot3), 0.0},
                        {std::sin(poni.rot3), std::cos(poni.rot3), 0.0},
                        {0.0, 0.0, 1.0}};
  double tmp[3][3];
  // np.dot(np.dot(rot3, rot2), rot1)
  matrix_multiplication(tmp, rot3, rot2);
  matrix_multiplication(rot, tmp, rot1);
  return;
}

void generate_matrix(const Poni& poni,
                     const py::array_t<double>& pixel_corners,
                     size_t n_splitting, 
                     std::vector<RListMatrix>& segments,
                     const int8_t* mask,
                     const Unit& output_unit,
                     const size_t nradial_bins, const double* radial_bins,
                     const size_t nphi_bins, const double* phi_bins)
{
    double rot[3][3];
    rotation_matrix(rot,poni);
    
    // h, w, corner index [A, B, C, D], coordinates [z, y, x]
    // A D
    // B C
    auto pc = pixel_corners.unchecked<4>();
    auto shape = pixel_corners.shape();
    
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<static_cast<size_t>(shape[0]); i++)
      {
        const size_t rank = omp_get_thread_num();
        for (size_t j=0; j<static_cast<size_t>(shape[1]); j++)
	  {
	    size_t pixel_index = i*static_cast<size_t>(shape[1])+j;
	    if (mask[pixel_index])
	      {
		continue;
	      }
            
            double A1 = pc(i,j,0,1);
            double A2 = pc(i,j,0,2);
            double A3 = pc(i,j,0,0);
            
            double BA1 = pc(i,j,1,1) - A1;
            double BA2 = pc(i,j,1,2) - A2;
            double BA3 = pc(i,j,1,0) - A3;
            
            double DA1 = pc(i,j,3,1) - A1;
            double DA2 = pc(i,j,3,2) - A2;
            double DA3 = pc(i,j,3,0) - A3;
            
            for (size_t k=0; k<n_splitting; k++)
	      {
		double delta1 = (k + 0.5) / static_cast<double>(n_splitting);
                for (size_t l=0; l<n_splitting; l++)
		  {
                    double delta2 =(l+0.5)/static_cast<double>(n_splitting);
                    double p[] =
		      {
                        A1 + delta1 * BA1 + delta2 * DA1 - poni.poni1,
                        A2 + delta1 * BA2 + delta2 * DA2 - poni.poni2,
                        A3 + delta1 * BA3 + delta2 * DA3 + poni.dist
		      };
                    double pos[3];
                    dot(pos, rot, p);
                    
                    const double r = std::sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
                    const double tth = std::atan2(r, pos[2]);
                    
                    double radial_coord = 0.0;
                    switch(output_unit)
		      {
                        case Unit::q:
			  // 4pi / lambda sin(2theta / 2) in A-1
			  radial_coord = 4.0e-10 * M_PI / poni.wavelength *
			    std::sin(0.5*tth);
			  break;

                        case Unit::tth:
                            // convert rad to deg
                            radial_coord=180.0/M_PI*tth;
                            break;
		      }

                    auto lower = std::lower_bound(radial_bins, 
                                                  radial_bins+nradial_bins+1, 
                                                  radial_coord);
                    const long int radial_index =
		      std::distance(radial_bins,lower)-1;
                    if ((radial_index < 0) ||
			(radial_index >= static_cast<long int>(nradial_bins)))
		      {
                        continue;
		      }
                    size_t bin_index = static_cast<size_t>(radial_index);
                    
                    // 2D integration
                    if (phi_bins)
		      {
                        // convert atan2 from [-pi, pi] to [0, 360] degrees
                        const double phi = std::atan2(-pos[0], -pos[1])
			  / M_PI*180.0f + 180.0;
                        
                        auto lower = std::lower_bound(phi_bins, 
                                                      phi_bins+nphi_bins+1, 
                                                      phi);
                        const long int phi_index =
			  std::distance(phi_bins, lower)-1;
                        if ((phi_index < 0) ||
			    (phi_index >= static_cast<long int>(nphi_bins)))
			  {
                            continue;
			  }
                        bin_index += static_cast<size_t>(phi_index)*
			  nradial_bins;
                    }
                    
                    auto& row = segments[rank].rows[bin_index];
                    // sum duplicate entries
                    if (row.size() > 0 && (row.back().col == pixel_index))
		      {
                        row.back().value +=
			  1.0/static_cast<double>(n_splitting * n_splitting);
		      }
                    else
		      {
                        row.emplace_back
			  (pixel_index,1.0/static_cast<double>
			   (n_splitting * n_splitting));
		      }
                }
            }
        }
    }
}

Sparse::Sparse(py::object py_poni,
               py::array_t<double> pixel_corners,
               const size_t n_splitting, 
               py::array_t<int8_t> mask,
               const std::string& unit,
               py::array_t<double, py::array::c_style | py::array::forcecast> radial_bins,
               std::optional<py::array_t<double, py::array::c_style | py::array::forcecast> > phi_bins)
{
    Poni poni;
    poni.dist = py_poni.attr("dist").cast<double>();
    poni.poni1 = py_poni.attr("poni1").cast<double>();
    poni.poni2 = py_poni.attr("poni2").cast<double>();
    poni.rot1 = py_poni.attr("rot1").cast<double>();
    poni.rot2 = py_poni.attr("rot2").cast<double>();
    poni.rot3 = py_poni.attr("rot3").cast<double>();
    poni.wavelength = py_poni.attr("wavelength").cast<double>();
    
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
    double* phi_data;
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
    generate_matrix(poni, 
                    pixel_corners, 
                    n_splitting, 
                    segments, 
                    mask.data(),
                    output_unit,
                    nradial_bins, radial_bins.data(),
                    nphi_bins, phi_data);
    
    tocsr(segments, nrows, col_idx, row_ptr, values);
}

Sparse::Sparse(std::vector<size_t>&& c,
               std::vector<size_t>&& r,
               std::vector<double>&& v,
               std::vector<double>&& vc,
               std::vector<double>&& vc2) : col_idx(c), row_ptr(r), values(v), values_corrected(vc), values_corrected2(vc2)
{
}

void Sparse::set_correction(py::array_t<double> corrections)
{
    values_corrected.resize(values.size());
    values_corrected2.resize(values.size());
    const double* cdata = corrections.data();
    size_t nrows = row_ptr.size() - 1;
    for (size_t i=0; i<nrows; i++)
      {
        for (size_t j=row_ptr[i]; j<row_ptr[i+1]; j++)
	  {
            // values_corrected = c / (solidangle * polarization)
            values_corrected[j] = values[j] / cdata[col_idx[j]];
            values_corrected2[j] =
	      values[j] * values[j] / (cdata[col_idx[j]] * cdata[col_idx[j]]);
	  }
      }
}


// sparse matrix vector multiplication A * x with sparse matrix A and vector x
template <typename T>
void _spmv(const size_t nrows,
	   const std::vector<size_t>& col_idx, 
           const std::vector<size_t>& row_ptr, 
           const std::vector<double>& values, 
           double* b, 
           const T* x)
{

    for (size_t i=0; i<nrows; i++)
      {
	double sum = 0.0;
	for (size_t j=row_ptr[i]; j<row_ptr[i+1]; j++)
	  sum += values[j] * static_cast<double>(x[col_idx[j]]);

      b[i] = sum;
    }
}

// sparse matrix vector multiplication A * x with sparse matrix A and vector x
template <typename T>
void
_spmvPair(const size_t nrows,
	  const std::vector<size_t>& col_idx, 
	  const std::vector<size_t>& row_ptr, 
	  const std::vector<double>& values,
	  const std::vector<double>& values2, 
	  double* b,
	  double* bErr, 
	  const T* x)
{
    for (size_t i=0; i<nrows; i++)
      {
	double sumA(0.0);
	double sumB(0.0);
	for (size_t j=row_ptr[i]; j<row_ptr[i+1]; j++)
	  {
	    sumA += values[j] * static_cast<double>(x[col_idx[j]]);
	    sumB += values2[j] * static_cast<double>(x[col_idx[j]]);
	  }
      b[i] = sumA;
      bErr[i] = sumB;
    }
}

std::tuple<py::array_t<double>,py::array_t<double>>
spmvPair(const std::vector<size_t>& col_idx, 
	 const std::vector<size_t>& row_ptr, 
	 const std::vector<double>& values,
	 const std::vector<double>& values2,
	 const py::array& x)
{
  //  static double sum(0.0);
  //  auto aTime =cppm::timer::Clock::now();

  const size_t nrows = row_ptr.size()-1;
  py::array_t<double,py::array::c_style> b(nrows);
  py::array_t<double,py::array::c_style> bErr(nrows);
  
  if (py::isinstance<py::array_t<uint8_t>>(x)) {
    py::gil_scoped_release release;
    _spmvPair(nrows, col_idx, row_ptr, values, values2,
	  b.mutable_data(), bErr.mutable_data(), (uint8_t*)x.data());
  }
  else if (py::isinstance<py::array_t<uint16_t>>(x)) {
    py::gil_scoped_release release;
    _spmvPair(nrows, col_idx, row_ptr, values, values2,
	      b.mutable_data(),bErr.mutable_data(), (uint16_t*)x.data());
  }
  else if (py::isinstance<py::array_t<uint32_t>>(x))
    {
      py::gil_scoped_release release;
      _spmvPair(nrows, col_idx, row_ptr, values, values2,
		b.mutable_data(),bErr.mutable_data(), (uint32_t*)x.data());
    }
  else if (py::isinstance<py::array_t<int8_t>>(x))
    {
      py::gil_scoped_release release;
      _spmvPair(nrows, col_idx, row_ptr, values, values2,
	    b.mutable_data(), bErr.mutable_data(), (int8_t*)x.data());
    }
  else if (py::isinstance<py::array_t<int16_t>>(x))
    {
      py::gil_scoped_release release;
      _spmvPair(nrows, col_idx, row_ptr, values, values2,
	    b.mutable_data(),bErr.mutable_data(), (int16_t*)x.data());
    }
  else if (py::isinstance<py::array_t<int32_t>>(x)) {
    py::gil_scoped_release release;
    _spmvPair(nrows, col_idx, row_ptr, values, values2,
	      b.mutable_data(),
	      bErr.mutable_data(),(int32_t*)x.data());
  }
  else if (py::isinstance<py::array_t<float>>(x))
    {
      py::gil_scoped_release release;
      _spmvPair(nrows, col_idx, row_ptr, values, values2,
		b.mutable_data(),
		bErr.mutable_data(),(float*)x.data());
    }
  else if (py::isinstance<py::array_t<double>>(x))
    {
      py::gil_scoped_release release;
      _spmvPair(nrows, col_idx, row_ptr, values,values2,
	    b.mutable_data(),
	    bErr.mutable_data(),(double*)x.data());
  }
  else {
    throw std::runtime_error("data dtype not supported");
  }
  
  // auto bTime =cppm::timer::Clock::now();
  // sum+=cppm::timer::Seconds(bTime-aTime).count();
  // printf("TimePAIR == %g %g\n",sum*1000.0,
  // 	 1000.0*cppm::timer::Seconds(bTime-aTime).count());

  return std::make_tuple(b,bErr);
}



py::array_t<double>
spmv(const std::vector<size_t>& col_idx, 
     const std::vector<size_t>& row_ptr, 
     const std::vector<double>& values,
     const py::array& x)
{
  // static double sum(0.0);
  // auto aTime =cppm::timer::Clock::now();

  int nrows = row_ptr.size() - 1;
  py::array_t<double,  py::array::c_style> b(nrows);
  
  if (py::isinstance<py::array_t<uint8_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint8_t*)x.data());
  }
  else if (py::isinstance<py::array_t<uint16_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint16_t*)x.data());
  }
  else if (py::isinstance<py::array_t<uint32_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (uint32_t*)x.data());
  }
  else if (py::isinstance<py::array_t<int8_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int8_t*)x.data());
  }
  else if (py::isinstance<py::array_t<int16_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int16_t*)x.data());
  }
  else if (py::isinstance<py::array_t<int32_t>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (int32_t*)x.data());
  }
  else if (py::isinstance<py::array_t<float>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (float*)x.data());
  }
  else if (py::isinstance<py::array_t<double>>(x)) {
    py::gil_scoped_release release;
    _spmv(nrows, col_idx, row_ptr, values, b.mutable_data(), (double*)x.data());
  }
  else {
    throw std::runtime_error("data dtype not supported");
  }

  // auto bTime =cppm::timer::Clock::now();
  // sum+=cppm::timer::Seconds(bTime-aTime).count();
  // printf("Time == %g %g\n",sum*1000.0,
  // 	 1000.0*cppm::timer::Seconds(bTime-aTime).count());

  return b;
}

py::array_t<double> Sparse::spmv(py::array x)
{
    return ::spmv(col_idx, row_ptr, values, x);
}

py::array_t<double> Sparse::spmv_corrected(py::array x)
{
    return ::spmv(col_idx, row_ptr, values_corrected, x);
}

py::array_t<double> Sparse::spmv_corrected2(py::array x)
{
    return ::spmv(col_idx, row_ptr, values_corrected2, x);
}

std::tuple<py::array_t<double>,py::array_t<double>>
Sparse::spmv_correctedPair(py::array x)
{
  return ::spmvPair(col_idx, row_ptr, values_corrected,values_corrected2, x);
}

PYBIND11_MODULE(_azint, m) {
    py::class_<Sparse>(m, "Sparse")
        .def(py::init<py::object, py::array_t<float>, int, py::array_t<int8_t>, std::string, py::array_t<float>, std::optional<py::array_t<float> > >())
        .def("set_correction", &Sparse::set_correction)
        .def("spmv", &Sparse::spmv)
        .def("spmv_corrected", &Sparse::spmv_corrected)
        .def("spmv_corrected2", &Sparse::spmv_corrected2)
        .def("spmv_correctedPair", &Sparse::spmv_correctedPair)
        .def(py::pickle(
            [](const Sparse &s) {
                return py::make_tuple(s.col_idx, s.row_ptr, s.values, s.values_corrected, s.values_corrected2);
            },
            [](py::tuple t) {
                Sparse s(std::move(t[0].cast<std::vector<size_t> >()),
                         std::move(t[1].cast<std::vector<size_t> >()),
                         std::move(t[2].cast<std::vector<double> >()),
                         std::move(t[3].cast<std::vector<double> >()),
                         std::move(t[4].cast<std::vector<double> >()));
                return s;
            }
        ));
}
