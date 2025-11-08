#include <stdexcept>

#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m)
{
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<const std::vector<std::vector<double>> &>(),
             "Initializes a matrix from a list of lists (2D Python list).")
        .def("__copy__", [](const Matrix &self)
             { return Matrix(self); })
        .def("__deepcopy__", [](const Matrix &self, py::dict)
             { return Matrix(self); })
        .def("__repr__", [](const Matrix &self)
             { return "<Matrix " + std::to_string(self.get_rows()) + "x" + std::to_string(self.get_cols()) + ">"; })
        .def("__str__", [](const Matrix &self)
             { return self.to_string(); })
        .def("__eq__", &Matrix::operator==)
        .def_property_readonly("nrow", &Matrix::get_rows)
        .def_property_readonly("ncol", &Matrix::get_cols)
        .def_property_readonly("size", [](const Matrix &self)
                               { return self.get_rows() * self.get_cols(); })
        .def("__getitem__", [](const Matrix &mat, std::pair<size_t, size_t> idx)
             { return mat(idx.first, idx.second); })
        .def("__setitem__", [](Matrix &mat, std::pair<size_t, size_t> idx, double value)
             { mat(idx.first, idx.second) = value; });

    m.def("multiply_naive", &multiply_naive, "Naive matrix multiplication");
    m.def("multiply_tile", &multiply_tile,
          "Tiled matrix multiplication with optional tile size (default 64)",
          py::arg("A"),
          py::arg("B"),
          py::arg("tile_size") = 64);
    m.def("multiply_mkl", &multiply_mkl, "MKL-based matrix multiplication");

    // Memory tracking functions
    m.def("bytes", []()
          { return MemoryTracker::total_bytes; }, "Get current bytes in use");
    m.def("allocated", []()
          { return MemoryTracker::total_allocated; }, "Get total allocated bytes");
    m.def("deallocated", []()
          { return MemoryTracker::total_deallocated; }, "Get total deallocated bytes");
}

// Naive matrix multiplication
Matrix multiply_naive(const Matrix &A, const Matrix &B)
{
    if (A.get_cols() != B.get_rows())
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix C(A.get_rows(), B.get_cols());

    for (size_t i = 0; i < A.get_rows(); ++i)
    {
        for (size_t j = 0; j < B.get_cols(); ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < A.get_cols(); ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

// Tiled matrix multiplication
Matrix multiply_tile(const Matrix &A, const Matrix &B, size_t tile_size)
{
    if (A.get_cols() != B.get_rows())
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix C(A.get_rows(), B.get_cols());

    for (size_t i = 0; i < A.get_rows(); i += tile_size)
    {
        for (size_t j = 0; j < B.get_cols(); j += tile_size)
        {
            for (size_t k = 0; k < A.get_cols(); k += tile_size)
            {
                // Multiply the tiles
                for (size_t ii = i; ii < std::min(i + tile_size, A.get_rows()); ++ii)
                {
                    for (size_t jj = j; jj < std::min(j + tile_size, B.get_cols()); ++jj)
                    {
                        double sum = C(ii, jj);
                        for (size_t kk = k; kk < std::min(k + tile_size, A.get_cols()); ++kk)
                        {
                            sum += A(ii, kk) * B(kk, jj);
                        }
                        C(ii, jj) = sum;
                    }
                }
            }
        }
    }

    return C;
}

// MKL-based matrix multiplication
Matrix multiply_mkl(const Matrix &A, const Matrix &B)
{
    if (A.get_cols() != B.get_rows())
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix C(A.get_rows(), B.get_cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A.get_rows(), B.get_cols(), A.get_cols(),
                1.0,
                A.data(), A.get_cols(),
                B.data(), B.get_cols(),
                0.0,
                C.data(), C.get_cols());

    return C;
}