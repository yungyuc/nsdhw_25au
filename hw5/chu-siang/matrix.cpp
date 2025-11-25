#include "matrix.hpp"

#include <algorithm> // std::min
#include <stdexcept>

#ifdef HAVE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#endif

namespace {

// shape check: A(M,K) * B(K,N)
inline void check_mul_dims(std::size_t a_m, std::size_t a_k,
                           std::size_t b_m, std::size_t b_n)
{
    if (a_k != b_m) {
        throw std::runtime_error("shape mismatch: A(M,K) x B(K,N) required");
    }
    (void)a_m;
    (void)b_n;
}

} // unnamed namespace

Matrix multiply_naive(const Matrix& A, const Matrix& B)
{
    const std::size_t M  = A.nrow();
    const std::size_t K  = A.ncol();
    const std::size_t K2 = B.nrow();
    const std::size_t N  = B.ncol();

    check_mul_dims(M, K, K2, N);

    Matrix C(M, N);

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

Matrix multiply_tile(const Matrix& A, const Matrix& B, int tsize)
{
    if (tsize <= 0) {
        throw std::runtime_error("tsize must be positive");
    }
    const std::size_t T = static_cast<std::size_t>(tsize);

    const std::size_t M  = A.nrow_;
    const std::size_t K  = A.ncol_;
    const std::size_t K2 = B.nrow_;
    const std::size_t N  = B.ncol_;

    check_mul_dims(M, K, K2, N);

    Matrix C(M, N);

    for (std::size_t ii = 0; ii < M; ii += T) {
        for (std::size_t kk = 0; kk < K; kk += T) {
            for (std::size_t jj = 0; jj < N; jj += T) {

                const std::size_t iimax = std::min(ii + T, M);
                const std::size_t kkmax = std::min(kk + T, K);
                const std::size_t jjmax = std::min(jj + T, N);

                for (std::size_t i = ii; i < iimax; ++i) {
                    double* c_row_base = &C.buf_[i * N + jj];

                    for (std::size_t k = kk; k < kkmax; ++k) {
                        const double aik = A.buf_[i * K + k];
                        if (aik == 0.0) {
                            continue; 
                        }

                        const double* bptr = &B.buf_[k * N + jj];
                        double*       cptr = c_row_base;

                        for (std::size_t j = jj; j < jjmax; ++j) {
                            *cptr++ += aik * (*bptr++);
                        }
                    }
                }

            }
        }
    }

    return C;
}

Matrix multiply_mkl(const Matrix& A, const Matrix& B)
{
    const std::size_t M  = A.nrow();
    const std::size_t K  = A.ncol();
    const std::size_t K2 = B.nrow();
    const std::size_t N  = B.ncol();

    check_mul_dims(M, K, K2, N);

#ifdef HAVE_MKL
    Matrix C(M, N);

    const double alpha = 1.0;
    const double beta  = 0.0;

    // row-major: lda=K, ldb=N, ldc=N
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        static_cast<MKL_INT>(M),
        static_cast<MKL_INT>(N),
        static_cast<MKL_INT>(K),
        alpha,
        A.data(), static_cast<MKL_INT>(K),
        B.data(), static_cast<MKL_INT>(N),
        beta,
        C.data(), static_cast<MKL_INT>(N)
    );

    return C;
#else
    return multiply_naive(A, B);
#endif
}
