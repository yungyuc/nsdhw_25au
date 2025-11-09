#include "Matrix.h"
#include "CustomAllocator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mkl.h>
#include <algorithm>
#include <chrono>

namespace py = pybind11;

// Naive matrix multiplication: O(n^3)
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    size_t m = A.rows();
    size_t n = A.cols();
    size_t p = B.cols();
    
    Matrix C(m, p);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    
    return C;
}

// Tiled matrix multiplication for better cache performance
Matrix multiply_tile(const Matrix& A, const Matrix& B, size_t tile_size) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    size_t m = A.rows();
    size_t n = A.cols();
    size_t p = B.cols();
    
    Matrix C(m, p);
    
    // Tiled multiplication
    for (size_t i = 0; i < m; i += tile_size) {
        for (size_t j = 0; j < p; j += tile_size) {
            for (size_t k = 0; k < n; k += tile_size) {
                // Compute the tile boundaries
                size_t i_end = std::min(i + tile_size, m);
                size_t j_end = std::min(j + tile_size, p);
                size_t k_end = std::min(k + tile_size, n);
                
                // Multiply the tiles
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; ++jj) {
                        double sum = C(ii, jj);
                        for (size_t kk = k; kk < k_end; ++kk) {
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

// Matrix multiplication using MKL DGEMM
Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    size_t m = A.rows();
    size_t n = A.cols();
    size_t p = B.cols();
    
    Matrix C(m, p);
    
    // DGEMM parameters
    // C = alpha * A * B + beta * C
    double alpha = 1.0;
    double beta = 0.0;
    
    cblas_dgemm(
        CblasRowMajor,      // Layout
        CblasNoTrans,       // transa
        CblasNoTrans,       // transb
        m,                  // M
        p,                  // N
        n,                  // K
        alpha,              // alpha
        A.data(),           // A
        n,                  // lda
        B.data(),           // B
        p,                  // ldb
        beta,               // beta
        C.data(),           // C
        p                   // ldc
    );
    
    return C;
}

// Python bindings
PYBIND11_MODULE(_matrix, m) {
    m.doc() = "Matrix multiplication module with memory tracking";
    
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, double>())
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols)
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def("__call__", [](const Matrix& m, size_t i, size_t j) {
            return m(i, j);
        })
        .def("__getitem__", [](const Matrix& m, std::pair<size_t, size_t> idx) {
            return m(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& m, std::pair<size_t, size_t> idx, double val) {
            m(idx.first, idx.second) = val;
        })
        .def("set", [](Matrix& m, size_t i, size_t j, double val) {
            m(i, j) = val;
        })
        .def("fill", &Matrix::fill)
        .def("equals", &Matrix::equals, py::arg("other"), py::arg("tol") = 1e-10)
        .def("__eq__", [](const Matrix& a, const Matrix& b) {
            return a.equals(b, 1e-10);
        })
        .def("__repr__", [](const Matrix& m) {
            return "Matrix(" + std::to_string(m.rows()) + "x" + std::to_string(m.cols()) + ")";
        });
    
    m.def("multiply_naive", &multiply_naive, "Naive matrix multiplication");
    m.def("multiply_tile", &multiply_tile, "Tiled matrix multiplication", 
          py::arg("A"), py::arg("B"), py::arg("tile_size") = 64);
    m.def("multiply_mkl", &multiply_mkl, "MKL DGEMM matrix multiplication");
    
    // Memory tracking functions
    m.def("bytes", &MemoryTracker::bytes, 
          "Return the current number of bytes in use");
    m.def("allocated", &MemoryTracker::allocated, 
          "Return the total number of bytes allocated");
    m.def("deallocated", &MemoryTracker::deallocated, 
          "Return the total number of bytes deallocated");
    m.def("reset_memory_tracker", &MemoryTracker::reset,
          "Reset memory tracking counters (for testing)");
}
