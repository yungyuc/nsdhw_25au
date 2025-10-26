#include <iostream>
#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <limits>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>




// Handle different BLAS library includes
#if defined(USE_MKL)
  #include <mkl_cblas.h>
  #define CBLAS_AVAILABLE
#elif defined(__APPLE__)
  #include <Accelerate/Accelerate.h>
  #define CBLAS_AVAILABLE
#else
  // Try to include cblas.h for OpenBLAS or other implementations
  #ifdef __has_include
    #if __has_include(<cblas.h>)
      #include <cblas.h>
      #define CBLAS_AVAILABLE
    #elif __has_include(<openblas/cblas.h>)
      #include <openblas/cblas.h>
      #define CBLAS_AVAILABLE
    #endif
  #endif

  // If no CBLAS found but we still need the constants for compilation
  #ifndef CBLAS_AVAILABLE
    #define CblasRowMajor 101
    #define CblasNoTrans 111
    // Don't declare cblas_dgemm here - we'll handle it in the function
  #endif
#endif

namespace py = pybind11;

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t r, std::size_t c, double init = 0.0)
        : rows_(r), cols_(c), data_(r * c, init) {}

    static Matrix Random(std::size_t r, std::size_t c,
                         unsigned seed = 0xC0FFEEu,
                         double lo = -1.0, double hi = 1.0) {
        Matrix m(r, c);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        for (double &x : m.data_) x = dist(rng);
        return m;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    // Element access (row-major)
    double& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    const double& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    // Helpful for unit tests
    bool approx_equal(const Matrix& other, double tol = 1e-9) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (std::size_t i = 0; i < size(); ++i) {
            double a = data_[i], b = other.data_[i];
            double diff = std::abs(a - b);
            double scale = std::max({1.0, std::abs(a), std::abs(b)});
            if (diff > tol * scale) return false;
        }
        return true;
    }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

static inline void check_mul_dims(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Dimension mismatch: A.cols()!=B.rows().");
    }
}

Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    check_mul_dims(A, B);
    Matrix C(A.rows(), B.cols(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols(); ++k)
                sum += A(i,k) * B(k,j);
            C(i,j) = sum;
        }
    return C;
}

Matrix multiply_tile(const Matrix& A, const Matrix& B, int block) {
    check_mul_dims(A, B);
    if (block <= 0) block = 16;

    Matrix C(A.rows(), B.cols(), 0.0);

    for (size_t ii = 0; ii < A.rows(); ii += (size_t)block) {
        for (size_t jj = 0; jj < B.cols(); jj += (size_t)block) {
            for (size_t kk = 0; kk < A.cols(); kk += (size_t)block) {

                const size_t i_max = std::min(ii + (size_t)block, A.rows());
                const size_t j_max = std::min(jj + (size_t)block, B.cols());
                const size_t k_max = std::min(kk + (size_t)block, A.cols());

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const double aik = A(i, k);
                        for (size_t j = jj; j < j_max; ++j) {
                            C(i, j) += aik * B(k, j);
                        }
                    }
                }
            }
        }
    }
    return C;
}

Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
    check_mul_dims(A, B);
    Matrix C(A.rows(), B.cols(), 0.0);

    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A.data(), K,
                B.data(), N,
                0.0,
                C.data(), N);
    return C;
}


PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>(), py::arg("nrow"), py::arg("ncol"))
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols)
        .def("__getitem__", [](const Matrix& m, std::pair<std::size_t,std::size_t> ij){
            return m(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& m, std::pair<std::size_t,std::size_t> ij, double v){
            m(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& a, const Matrix& b){ return a.approx_equal(b, 1e-9); })
        .def("__ne__", [](const Matrix& a, const Matrix& b){ return !a.approx_equal(b, 1e-9); })
        .def("approx_equal", &Matrix::approx_equal, py::arg("other"), py::arg("tol") = 1e-9);

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile",
          [](const Matrix& A, const Matrix& B, int block){ return multiply_tile(A, B, block); },
          py::arg("A"), py::arg("B"), py::arg("block") = 64);
    m.def("multiply_mkl", &multiply_mkl);
}