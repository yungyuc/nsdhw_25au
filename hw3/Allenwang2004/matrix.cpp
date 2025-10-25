#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include <pybind11/pybind11.h>

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

class Matrix {
public:
    Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    // (row, col) in row-major
    inline double& operator()(int r, int c) { return data_[r * cols_ + c]; }
    inline const double& operator()(int r, int c) const { return data_[r * cols_ + c]; }

    inline int rows() const { return rows_; }
    inline int cols() const { return cols_; }
    inline double* data() { return data_.data(); }
    inline const double* data() const { return data_.data(); }

    // Equality operator
    bool operator==(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                if (std::abs((*this)(i, j) - other(i, j)) > 1e-10) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    int rows_, cols_;
    std::vector<double> data_;  // contiguous buffer
};

// Naive i-j-k triple loop
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::invalid_argument("Incompatible dimensions");
    Matrix C(A.rows(), B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols(); ++k)
                sum += A(i, k) * B(k, j);
            C(i, j) = sum;
        }
    return C;
}

// Tiled (blocked) version
Matrix multiply_tile(const Matrix& A, const Matrix& B, int T) {
    if (A.cols() != B.rows()) throw std::invalid_argument("Incompatible dimensions");
    Matrix C(A.rows(), B.cols());
    const int M = A.rows(), N = B.cols(), K = A.cols();

    for (int ii = 0; ii < M; ii += T)
        for (int kk = 0; kk < K; kk += T)
            for (int jj = 0; jj < N; jj += T) {
                const int iimax = std::min(ii + T, M);
                const int kkmax = std::min(kk + T, K);
                const int jjmax = std::min(jj + T, N);
                for (int i = ii; i < iimax; ++i)
                    for (int k = kk; k < kkmax; ++k) {
                        const double aik = A(i, k);
                        for (int j = jj; j < jjmax; ++j)
                            C(i, j) += aik * B(k, j);
                    }
            }
    return C;
}

// DGEMM using CBLAS (row-major) or fallback to tiled if not available
Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::invalid_argument("Incompatible dimensions");
    
#ifdef CBLAS_AVAILABLE
    Matrix C(A.rows(), B.cols());

    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();
    const double alpha = 1.0, beta = 0.0;

    // C = alpha * A * B + beta * C
    cblas_dgemm(
        CblasRowMajor,     // our buffers are row-major
        CblasNoTrans,      // op(A)
        CblasNoTrans,      // op(B)
        M, N, K,
        alpha,
        A.data(), K,       // lda = K for row-major
        B.data(), N,       // ldb = N for row-major
        beta,
        C.data(), N        // ldc = N for row-major
    );
    return C;
#else
    // No CBLAS available, fallback to tiled implementation
    return multiply_tile(A, B, 64);
#endif
}

PYBIND11_MODULE(_matrix, m) {
    pybind11::class_<Matrix>(m, "Matrix")
        .def(pybind11::init<int, int>())
        .def("__getitem__", [](const Matrix &mat, std::pair<int, int> idx) {
            return mat(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix &mat, std::pair<int, int> idx, double value) {
            mat(idx.first, idx.second) = value;
        })
        .def("__eq__", &Matrix::operator==)
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols);

    m.def("multiply_naive", &multiply_naive, "Naive matrix multiplication");
    m.def("multiply_tile", &multiply_tile, "Tiled matrix multiplication", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("T"));
    m.def("multiply_mkl", &multiply_mkl, "MKL matrix multiplication");
}

