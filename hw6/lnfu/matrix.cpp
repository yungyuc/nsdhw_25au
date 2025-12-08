#include <stdexcept>

#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include "matrix.hpp"

// Naive matrix multiplication
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
  if (A.ncol() != B.nrow()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for multiplication");
  }

  Matrix C(A.nrow(), B.ncol());

  for (size_t i = 0; i < A.nrow(); ++i) {
    for (size_t j = 0; j < B.ncol(); ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < A.ncol(); ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }

  return C;
}

// Tiled matrix multiplication
Matrix multiply_tile(const Matrix& A, const Matrix& B, size_t tile_size) {
  if (A.ncol() != B.nrow()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for multiplication");
  }

  Matrix C(A.nrow(), B.ncol());

  for (size_t i = 0; i < A.nrow(); i += tile_size) {
    for (size_t j = 0; j < B.ncol(); j += tile_size) {
      for (size_t k = 0; k < A.ncol(); k += tile_size) {
        // Multiply the tiles
        for (size_t ii = i; ii < std::min(i + tile_size, A.nrow()); ++ii) {
          for (size_t jj = j; jj < std::min(j + tile_size, B.ncol());
               ++jj) {
            double sum = C(ii, jj);
            for (size_t kk = k; kk < std::min(k + tile_size, A.ncol());
                 ++kk) {
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
Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
  if (A.ncol() != B.nrow()) {
    throw std::invalid_argument(
        "Matrix dimensions incompatible for multiplication");
  }

  Matrix C(A.nrow(), B.ncol());

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.nrow(),
              B.ncol(), A.ncol(), 1.0, A.data(), A.ncol(), B.data(),
              B.ncol(), 0.0, C.data(), C.ncol());

  return C;
}
