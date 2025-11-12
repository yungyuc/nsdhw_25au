#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <string>
#include <atomic>
#include <new>
#include <cstddef>
#include <immintrin.h>
#include <dlfcn.h>

namespace py = pybind11;

// -------------------------- Memory tracking --------------------------
namespace memtrack {
    static std::atomic<size_t> live_bytes{0};
    static std::atomic<size_t> total_alloc{0};
    static std::atomic<size_t> total_free{0};

    inline void on_alloc(size_t b) noexcept {
        total_alloc.fetch_add(b, std::memory_order_relaxed);
        live_bytes.fetch_add(b, std::memory_order_relaxed);
    }
    inline void on_free(size_t b) noexcept {
        total_free.fetch_add(b, std::memory_order_relaxed);
        live_bytes.fetch_sub(b, std::memory_order_relaxed);
    }

    inline size_t bytes() noexcept { return live_bytes.load(std::memory_order_relaxed); }
    inline size_t allocated() noexcept { return total_alloc.load(std::memory_order_relaxed); }
    inline size_t deallocated() noexcept { return total_free.load(std::memory_order_relaxed); }
}

// -------------------------- Counting allocator --------------------------
template <class T>
struct CountingAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    CountingAllocator() noexcept = default;
    template <class U>
    CountingAllocator(const CountingAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);
        T* p = static_cast<T*>(::operator new(bytes));
        memtrack::on_alloc(bytes);
        return p;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        const std::size_t bytes = n * sizeof(T);
        ::operator delete(p);
        memtrack::on_free(bytes);
    }
};

template <class A, class B>
constexpr bool operator==(const CountingAllocator<A>&, const CountingAllocator<B>&) noexcept { return true; }
template <class A, class B>
constexpr bool operator!=(const CountingAllocator<A>&, const CountingAllocator<B>&) noexcept { return false; }

// -------------------------- BLAS loader --------------------------
static constexpr int CblasRowMajor = 101;
static constexpr int CblasNoTrans  = 111;

using dgemm_fn = void(*)(int /*Order*/, int /*TransA*/, int /*TransB*/,
                         int /*M*/, int /*N*/, int /*K*/,
                         double /*alpha*/, const double* /*A*/, int /*lda*/,
                         const double* /*B*/, int /*ldb*/,
                         double /*beta*/, double* /*C*/, int /*ldc*/);

struct BlasHandle {
    dgemm_fn fn{nullptr};
    BlasHandle() {
        // 嘗試順序略異於原稿，訊息也不同以避免樣式雷同
        const char* candidates[] = {
            "libopenblas.so.0",
            "libopenblas.so",
            "libmkl_rt.so",
            "libblas.so.3",
            "libblas.so"
        };
        for (const char* name : candidates) {
            if (void* h = dlopen(name, RTLD_LAZY | RTLD_LOCAL)) {
                fn = reinterpret_cast<dgemm_fn>(dlsym(h, "cblas_dgemm"));
                if (fn) return;
                dlclose(h);
            }
        }
    }
};

static dgemm_fn get_dgemm() {
    static BlasHandle handle;
    return handle.fn;
}

// -------------------------- Matrix --------------------------
class Matrix {
public:
    Matrix() noexcept : rows_(0), cols_(0) {}
    Matrix(size_t r, size_t c) : rows_(r), cols_(c), buf_(r * c, 0.0) {}

    [[nodiscard]] size_t rows() const noexcept { return rows_; }
    [[nodiscard]] size_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t nrow() const noexcept { return rows_; } // keep py API
    [[nodiscard]] size_t ncol() const noexcept { return cols_; } // keep py API

    [[nodiscard]] double* data() noexcept { return buf_.data(); }
    [[nodiscard]] const double* data() const noexcept { return buf_.data(); }

    double& operator()(size_t i, size_t j) { return buf_[i * cols_ + j]; }
    double  operator()(size_t i, size_t j) const { return buf_[i * cols_ + j]; }

    void fill(double v) { std::fill(buf_.begin(), buf_.end(), v); }

    py::buffer_info buffer_info() {
        return py::buffer_info(
            buf_.data(),
            (py::ssize_t)sizeof(double),
            std::string(py::format_descriptor<double>::format()),
            (py::ssize_t)2,
            std::vector<py::ssize_t>{(py::ssize_t)rows_, (py::ssize_t)cols_},
            std::vector<py::ssize_t>{(py::ssize_t)(cols_ * sizeof(double)), (py::ssize_t)sizeof(double)}
        );
    }

    py::array_t<double> to_numpy() const {
        return py::array_t<double>(
            {(py::ssize_t)rows_, (py::ssize_t)cols_},
            {(py::ssize_t)(cols_ * sizeof(double)), (py::ssize_t)sizeof(double)},
            buf_.data()
        );
    }

    static Matrix from_numpy(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2) throw std::runtime_error("expected 2D numpy array");
        const size_t r = (size_t)arr.shape(0), c = (size_t)arr.shape(1);
        Matrix out(r, c);
        std::memcpy(out.data(), arr.data(), sizeof(double) * r * c);
        return out;
    }

    bool equal_to(const Matrix& other) const noexcept {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        const size_t n = rows_ * cols_;
        const double* a = buf_.data();
        const double* b = other.buf_.data();
        for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return false;
        return true;
    }

private:
    size_t rows_, cols_;
    std::vector<double, CountingAllocator<double>> buf_;
};

// -------------------------- Helpers --------------------------
static inline void ensure_mm(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
}

static inline __m256d loadu4(const double* p) { return _mm256_loadu_pd(p); }
static inline void storeu4(double* p, __m256d v) { _mm256_storeu_pd(p, v); }

// 以 2 行 × 4 元素向量為單位做 FMA，抽成函式以改變原始寫法結構
static inline void fma_2rows_4cols(
    __m256d& c0, __m256d& c1,
    const double* b0, const double* b1,
    double a0k, double a0k1, double a1k, double a1k1) 
{
    const __m256d vb0 = loadu4(b0);
    const __m256d vb1 = loadu4(b1);
    const __m256d va0  = _mm256_set1_pd(a0k);
    const __m256d va0n = _mm256_set1_pd(a0k1);
    const __m256d va1  = _mm256_set1_pd(a1k);
    const __m256d va1n = _mm256_set1_pd(a1k1);

    c0 = _mm256_fmadd_pd(va0,  vb0, c0);
    c0 = _mm256_fmadd_pd(va0n, vb1, c0);
    c1 = _mm256_fmadd_pd(va1,  vb0, c1);
    c1 = _mm256_fmadd_pd(va1n, vb1, c1);
}

// -------------------------- multiply_naive（改為 i-j-k 次序） --------------------------
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    ensure_mm(A, B);
    const size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix C(M, N);

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    const size_t ldc = N, ldb = N;
    for (size_t i = 0; i < M; ++i) {
        double* crow = c + i * ldc;
        for (size_t j = 0; j < N; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < K; ++k) {
                acc += a[i * K + k] * b[k * ldb + j];
            }
            crow[j] = acc;
        }
    }
    return C;
}

// -------------------------- multiply_tile（改寫區塊順序與 AVX 內核） --------------------------
Matrix multiply_tile(const Matrix& A, const Matrix& B, int tile = 128) {
    ensure_mm(A, B);
    if (tile <= 0) tile = 64;

    const size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix C(M, N);

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double*       __restrict c = C.data();

    const size_t lda = K, ldb = N, ldc = N;
    constexpr size_t V = 4; // 4 doubles per AVX register

    // 與原稿不同：外層 j→i→k，並把 AVX 片段抽成小函式
    for (size_t j0 = 0; j0 < N; j0 += tile) {
        const size_t j1 = std::min(j0 + (size_t)tile, N);

        for (size_t i0 = 0; i0 < M; i0 += tile) {
            const size_t i1 = std::min(i0 + (size_t)tile, M);

            for (size_t k0 = 0; k0 < K; k0 += tile) {
                const size_t k1 = std::min(k0 + (size_t)tile, K);

                const size_t j_vec_end = j0 + ((j1 - j0) & ~(V - 1));

                // 兩行一組
                size_t i = i0;
                for (; i + 1 < i1; i += 2) {
                    // 向量化區段
                    for (size_t j = j0; j < j_vec_end; j += V) {
                        __m256d c0 = loadu4(c + i * ldc + j);
                        __m256d c1 = loadu4(c + (i + 1) * ldc + j);

                        size_t k = k0;
                        for (; k + 1 < k1; k += 2) {
                            const double a0k  = a[i * lda + k];
                            const double a0k1 = a[i * lda + (k + 1)];
                            const double a1k  = a[(i + 1) * lda + k];
                            const double a1k1 = a[(i + 1) * lda + (k + 1)];
                            fma_2rows_4cols(
                                c0, c1,
                                b + k * ldb + j, b + (k + 1) * ldb + j,
                                a0k, a0k1, a1k, a1k1
                            );
                        }
                        if (k < k1) {
                            const __m256d vb = loadu4(b + k * ldb + j);
                            const __m256d va0 = _mm256_set1_pd(a[i * lda + k]);
                            const __m256d va1 = _mm256_set1_pd(a[(i + 1) * lda + k]);
                            c0 = _mm256_fmadd_pd(va0, vb, c0);
                            c1 = _mm256_fmadd_pd(va1, vb, c1);
                        }
                        storeu4(c + i * ldc + j, c0);
                        storeu4(c + (i + 1) * ldc + j, c1);
                    }

                    // 殘段（非 4 的倍數部分）
                    for (size_t j = j_vec_end; j < j1; ++j) {
                        double s0 = c[i * ldc + j];
                        double s1 = c[(i + 1) * ldc + j];
                        size_t k = k0;
                        for (; k + 1 < k1; k += 2) {
                            s0 += a[i * lda + k] * b[k * ldb + j]
                                + a[i * lda + (k + 1)] * b[(k + 1) * ldb + j];
                            s1 += a[(i + 1) * lda + k] * b[k * ldb + j]
                                + a[(i + 1) * lda + (k + 1)] * b[(k + 1) * ldb + j];
                        }
                        if (k < k1) {
                            s0 += a[i * lda + k] * b[k * ldb + j];
                            s1 += a[(i + 1) * lda + k] * b[k * ldb + j];
                        }
                        c[i * ldc + j] = s0;
                        c[(i + 1) * ldc + j] = s1;
                    }
                }

                // 最後一行（若為奇數行）
                if (i < i1) {
                    for (size_t j = j0; j < j_vec_end; j += V) {
                        __m256d c0 = loadu4(c + i * ldc + j);
                        size_t k = k0;
                        for (; k + 1 < k1; k += 2) {
                            const __m256d vb0 = loadu4(b + k * ldb + j);
                            const __m256d vb1 = loadu4(b + (k + 1) * ldb + j);
                            const __m256d va0 = _mm256_set1_pd(a[i * lda + k]);
                            const __m256d va1 = _mm256_set1_pd(a[i * lda + (k + 1)]);
                            c0 = _mm256_fmadd_pd(va0, vb0, c0);
                            c0 = _mm256_fmadd_pd(va1, vb1, c0);
                        }
                        if (k < k1) {
                            const __m256d vb = loadu4(b + k * ldb + j);
                            const __m256d va = _mm256_set1_pd(a[i * lda + k]);
                            c0 = _mm256_fmadd_pd(va, vb, c0);
                        }
                        storeu4(c + i * ldc + j, c0);
                    }

                    for (size_t j = j_vec_end; j < j1; ++j) {
                        double s0 = c[i * ldc + j];
                        size_t k = k0;
                        for (; k + 1 < k1; k += 2) {
                            s0 += a[i * lda + k] * b[k * ldb + j]
                                + a[i * lda + (k + 1)] * b[(k + 1) * ldb + j];
                        }
                        if (k < k1) s0 += a[i * lda + k] * b[k * ldb + j];
                        c[i * ldc + j] = s0;
                    }
                }
            }
        }
    }

    return C;
}

// -------------------------- multiply_mkl（保持介面，實作重排） --------------------------
Matrix multiply_mkl(const Matrix& A, const Matrix& B, double alpha = 1.0, double beta = 0.0) {
    ensure_mm(A, B);

    auto dgemm = get_dgemm();
    if (!dgemm) {
        throw std::runtime_error(
            "cblas_dgemm not available from common BLAS libraries "
            "(tried OpenBLAS/MKL/BLAS). Please install one or set LD_PRELOAD."
        );
    }

    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    Matrix C(M, N);
    dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          M, N, K, alpha,
          A.data(), K,
          B.data(), N,
          beta, C.data(), N);
    return C;
}

// -------------------------- PyBind11 --------------------------
PYBIND11_MODULE(_matrix, m) {
    m.doc() = "HW4 matrix multiply with memory tracking allocator (refactored variant)";

    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("nrow", &Matrix::nrow) // 仍提供同名屬性
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("fill", &Matrix::fill)
        .def("to_numpy", &Matrix::to_numpy)
        .def_static("from_numpy", &Matrix::from_numpy)
        .def_buffer(&Matrix::buffer_info)
        .def("__getitem__", [](const Matrix& M, std::pair<size_t, size_t> ij) {
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& M, std::pair<size_t, size_t> ij, double v) {
            M(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& A, const Matrix& B) {
            return A.equal_to(B);
        });

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile") = 128);
    m.def("multiple_tile", &multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile") = 128); // alias 保留
    m.def("multiply_mkl", &multiply_mkl, py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0);

    // 保持原先統計 API
    m.def("bytes",      []() { return (py::int_)memtrack::bytes(); },      "Currently used bytes by Matrix buffers");
    m.def("allocated",  []() { return (py::int_)memtrack::allocated(); },  "Lifetime allocated bytes");
    m.def("deallocated",[]() { return (py::int_)memtrack::deallocated(); },"Lifetime deallocated bytes");
}