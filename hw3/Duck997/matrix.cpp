#include "matrix.hpp"
#include <cstring>
#include <cassert>

#if defined(__has_include)
#  if __has_include(<cblas.h>)
#    include <cblas.h>
#    define HAS_CBLAS 1
#  elif __has_include(<mkl.h>)
#    include <mkl.h>
#    define HAS_MKL 1
#  endif
#endif

Matrix multiply_naive(Matrix const &A, Matrix const &B) {
    if (A.cols() != B.rows()) throw std::invalid_argument("incompatible shapes");
    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t k = 0; k < A.cols(); ++k) {
            const double aik = A.at(i, k);
            for (size_t j = 0; j < B.cols(); ++j) {
                C.at(i, j) += aik * B.at(k, j);
            }
        }
    }
    return C;
}

Matrix multiple_tile(Matrix const &A, Matrix const &B, size_t tile) {
    if (A.cols() != B.rows()) throw std::invalid_argument("incompatible shapes");
    Matrix C(A.rows(), B.cols());
    const size_t n = A.rows();
    const size_t m = A.cols();
    const size_t p = B.cols();
    const size_t T = tile ? tile : 32;

    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();

    // Row-major indexing helpers
    auto Aidx = [m](size_t i, size_t k) { return i * m + k; };
    auto Bidx = [p](size_t k, size_t j) { return k * p + j; };
    auto Cidx = [p](size_t i, size_t j) { return i * p + j; };

    for (size_t ii = 0; ii < n; ii += T) {
        const size_t i_max = std::min(ii + T, n);
        for (size_t jj = 0; jj < p; jj += T) {
            const size_t j_max = std::min(jj + T, p);
            for (size_t kk = 0; kk < m; kk += T) {
                const size_t k_max = std::min(kk + T, m);
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const double aik = a[Aidx(i, k)];
                        const size_t bbase = Bidx(k, jj);
                        size_t cj = Cidx(i, jj);
                        for (size_t j = jj; j < j_max; ++j) {
                            c[cj++] += aik * b[bbase + (j - jj)];
                        }
                    }
                }
            }
        }
    }
    return C;
}

Matrix multiply_mkl(Matrix const &A, Matrix const &B) {
    if (A.cols() != B.rows()) throw std::invalid_argument("incompatible shapes");
    Matrix C(A.rows(), B.cols());

#if defined(HAS_CBLAS) || defined(HAS_MKL)
    const int M = static_cast<int>(A.rows());
    const int N = static_cast<int>(B.cols());
    const int K = static_cast<int>(A.cols());
    const double alpha = 1.0;
    const double beta = 0.0;
    // Row-major: C = alpha*A*B + beta*C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha,
                A.data(), K,
                B.data(), N,
                beta,
                C.data(), N);
#else
#ifndef NDEBUG
    assert(false && "BLAS (cblas/mkl) not available; falling back to naive multiply");
#endif
    C = multiply_naive(A, B);
#endif
    return C;
}

bool blas_available() {
#if defined(HAS_CBLAS) || defined(HAS_MKL)
    return true;
#else
    return false;
#endif
}


