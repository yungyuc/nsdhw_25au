// matrix.cpp
#include "matrix.hpp"

#include <algorithm>
#include <dlfcn.h>   // dlopen, dlsym
#include <stdexcept>

// ---------------- cblas_dgemm 設定 ----------------
// 這些常數跟 CBLAS 的 enum 一樣值，讓我們可以手動呼叫 cblas_dgemm。
static constexpr int CblasRowMajor = 101;
static constexpr int CblasNoTrans  = 111;

using dgemm_fn = void(*)(int Order, int TransA, int TransB,
                         int M, int N, int K,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb,
                         double beta, double *C, int ldc);

// 嘗試從常見 BLAS shared library 把 cblas_dgemm 抓出來
static dgemm_fn load_dgemm() {
    static dgemm_fn fn = nullptr;
    static bool tried = false;
    if (tried) return fn;
    tried = true;

    const char* libs[] = {
        "libmkl_rt.so",
        "libopenblas.so.0",
        "libopenblas.so",
        "libblas.so.3",
        "libblas.so"
    };

    for (const char* so : libs) {
        void* h = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
        if (!h) continue;
        fn = reinterpret_cast<dgemm_fn>(dlsym(h, "cblas_dgemm"));
        if (fn) return fn;
        dlclose(h);
    }
    return nullptr;
}

// ---------------- naive 版本（刻意寫得比較土） ----------------
Matrix multiply_naive(Matrix const& A, Matrix const& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("dimension mismatch");
    }

    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    Matrix C(M, N);

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    for (std::size_t i = 0; i < M; ++i) {
        const double* a_row = a + i * lda;
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            const double* b_col = b + j;  // 之後用 k*ldb + j
            for (std::size_t k = 0; k < K; ++k) {
                sum += a_row[k] * b_col[k * ldb];
            }
            c[i * ldc + j] = sum;
        }
    }

    return C;
}

// ---------------- tiled 版本 ----------------
Matrix multiply_tile(Matrix const& A, Matrix const& B, std::size_t tile) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("dimension mismatch");
    }

    if (tile == 0) {
        // 防呆：tile=0 就直接退回 naive
        return multiply_naive(A, B);
    }

    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    Matrix C(M, N);

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    const std::size_t lda = K;
    const std::size_t ldb = N;
    const std::size_t ldc = N;

    // 保險做一次清零（雖然建構子已經填 0 了）
    std::fill(c, c + M * N, 0.0);

    const std::size_t Ti = tile;
    const std::size_t Tj = tile;
    const std::size_t Tk = tile;

    // 三層 blocking：i0, k0, j0
    for (std::size_t i0 = 0; i0 < M; i0 += Ti) {
        const std::size_t i_max = std::min(i0 + Ti, M);

        for (std::size_t k0 = 0; k0 < K; k0 += Tk) {
            const std::size_t k_max = std::min(k0 + Tk, K);

            for (std::size_t j0 = 0; j0 < N; j0 += Tj) {
                const std::size_t j_max = std::min(j0 + Tj, N);

                // 真正 block 裡面的計算
                for (std::size_t i = i0; i < i_max; ++i) {
                    double*       c_row = c + i * ldc;
                    const double* a_row = a + i * lda;

                    for (std::size_t k = k0; k < k_max; ++k) {
                        const double  aik   = a_row[k];
                        const double* b_row = b + k * ldb;  // B 的一整列連續

                        // j 方向完全連續 → 對 b_row / c_row 都很友善
                        for (std::size_t j = j0; j < j_max; ++j) {
                            c_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    return C;
}

// ---------------- BLAS 版本（用 cblas_dgemm，如果找得到） ----------------
Matrix multiply_mkl(Matrix const& A, Matrix const& B) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("dimension mismatch");
    }

    auto dgemm = load_dgemm();
    if (!dgemm) {
        throw std::runtime_error(
            "BLAS library with cblas_dgemm not found. "
            "Install OpenBLAS/MKL or use LD_PRELOAD to provide it."
        );
    }

    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    Matrix C(A.rows(), B.cols());

    dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        /*alpha=*/1.0, A.data(), K,
        B.data(), N,
        /*beta=*/0.0, C.data(), N
    );

    return C;
}
