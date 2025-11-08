// matrix_ops.cpp
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
#include <dlfcn.h>   // for dlopen/dlsym

namespace py = pybind11;

// ===================== Global tracking =====================
namespace memtrack {
    static std::atomic<size_t> g_bytes{0};       // currently in-use by Matrix buffers
    static std::atomic<size_t> g_allocated{0};   // lifetime allocated
    static std::atomic<size_t> g_deallocated{0}; // lifetime deallocated

    inline void add_alloc(size_t b){
        g_allocated.fetch_add(b, std::memory_order_relaxed);
        g_bytes.fetch_add(b, std::memory_order_relaxed);
    }
    inline void add_dealloc(size_t b){
        g_deallocated.fetch_add(b, std::memory_order_relaxed);
        g_bytes.fetch_sub(b, std::memory_order_relaxed);
    }
    inline size_t bytes(){ return g_bytes.load(std::memory_order_relaxed); }
    inline size_t allocated(){ return g_allocated.load(std::memory_order_relaxed); }
    inline size_t deallocated(){ return g_deallocated.load(std::memory_order_relaxed); }
}

// ===================== Custom STL Allocator =====================
template <class T>
struct CountingAllocator {
    using value_type = T;

    CountingAllocator() noexcept {}
    template <class U>
    CountingAllocator(const CountingAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        T* p = static_cast<T*>(::operator new(bytes));
        memtrack::add_alloc(bytes);
        return p;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::size_t bytes = n * sizeof(T);
        ::operator delete(p);
        memtrack::add_dealloc(bytes);
    }

    using is_always_equal = std::true_type;
};

template <class T, class U>
bool operator==(const CountingAllocator<T>&, const CountingAllocator<U>&){ return true; }
template <class T, class U>
bool operator!=(const CountingAllocator<T>&, const CountingAllocator<U>&){ return false; }

// ---- cblas dgemm (row-major) enum constants ----
static constexpr int CblasRowMajor = 101;
static constexpr int CblasNoTrans  = 111;

// ---- Delayed loader for cblas_dgemm ----
using dgemm_fn = void(*)(int Order, int TransA, int TransB,
                         int M, int N, int K,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb,
                         double beta, double *C, int ldc);

static dgemm_fn load_dgemm() {
    static dgemm_fn fn = nullptr;
    static bool tried = false;
    if (tried) return fn;
    tried = true;

    // Common BLAS .so names to try
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

// ---------------- Matrix ----------------
class Matrix {
public:
    Matrix() : r_(0), c_(0) {}
    Matrix(size_t r, size_t c) : r_(r), c_(c), data_(r*c, 0.0) {}

    size_t rows() const { return r_; }
    size_t cols() const { return c_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    double& operator()(size_t i, size_t j) { return data_[i*c_ + j]; }
    double  operator()(size_t i, size_t j) const { return data_[i*c_ + j]; }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    py::buffer_info buffer_info() {
        return py::buffer_info(
            data_.data(),
            (py::ssize_t)sizeof(double),
            std::string(py::format_descriptor<double>::format()),
            (py::ssize_t)2,
            std::vector<py::ssize_t>{(py::ssize_t)r_, (py::ssize_t)c_},
            std::vector<py::ssize_t>{(py::ssize_t)(c_*sizeof(double)), (py::ssize_t)sizeof(double)}
        );
    }

    py::array_t<double> to_numpy() const {
        return py::array_t<double>(
            {(py::ssize_t)r_, (py::ssize_t)c_},
            {(py::ssize_t)(c_*sizeof(double)), (py::ssize_t)sizeof(double)},
            data_.data()
        );
    }

    static Matrix from_numpy(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2) throw std::runtime_error("expected 2D numpy array");
        size_t r = (size_t)arr.shape(0), c = (size_t)arr.shape(1);
        Matrix M(r, c);
        std::memcpy(M.data(), arr.data(), sizeof(double)*r*c);
        return M;
    }

private:
    size_t r_, c_;
    // >>> KEY CHANGE: use std::vector with our CountingAllocator <<<
    std::vector<double, CountingAllocator<double>> data_;
};

// --------- naive ----------
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
    const size_t M=A.rows(), K=A.cols(), N=B.cols();
    Matrix C(M,N);
    const double* a=A.data(); const double* b=B.data(); double* c=C.data();
    const size_t ldb=N, ldc=N;
    for (size_t i=0;i<M;++i){
        for(size_t k=0;k<K;++k){
            const double aik=a[i*K+k];
            const double* b_row=b+k*ldb;
            double* c_row=c+i*ldc;
            for(size_t j=0;j<N;++j){
                c_row[j]+=aik*b_row[j];
            }
        }
    }
    return C;
}

// --------- tiled (AVX2) ----------
Matrix multiply_tile(const Matrix& A, const Matrix& B, int tile=128){
    if (A.cols()!=B.rows()) throw std::runtime_error("dimension mismatch");
    if (tile<=0) tile=64;

    const size_t M=A.rows(), K=A.cols(), N=B.cols();
    Matrix C(M,N);

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();

    const size_t lda = K, ldb = N, ldc = N;
    const size_t vec_size = 4; // AVX2: 4 doubles

    for(size_t k0=0; k0<K; k0+=tile){
        const size_t k_max = std::min(k0 + (size_t)tile, K);

        for(size_t i0=0; i0<M; i0+=tile){
            const size_t i_max = std::min(i0 + (size_t)tile, M);

            for(size_t j0=0; j0<N; j0+=tile){
                const size_t j_max = std::min(j0 + (size_t)tile, N);

                for(size_t i=i0; i<i_max; i+=2){
                    const bool has_pair = (i + 1 < i_max);
                    const size_t i1 = i + 1;

                    const size_t j_vec_max = j0 + ((j_max - j0) & ~(vec_size - 1));

                    if (has_pair) {
                        for(size_t j=j0; j<j_vec_max; j+=vec_size){
                            __m256d c_acc0 = _mm256_loadu_pd(c + i*ldc  + j);
                            __m256d c_acc1 = _mm256_loadu_pd(c + i1*ldc + j);

                            size_t k = k0;
                            for(; k + 1 < k_max; k += 2){
                                const __m256d b0 = _mm256_loadu_pd(b + k*ldb     + j);
                                const __m256d b1 = _mm256_loadu_pd(b + (k+1)*ldb + j);

                                const __m256d a0  = _mm256_set1_pd(a[i*lda  + k]);
                                const __m256d a0n = _mm256_set1_pd(a[i*lda  + (k+1)]);
                                const __m256d a1  = _mm256_set1_pd(a[i1*lda + k]);
                                const __m256d a1n = _mm256_set1_pd(a[i1*lda + (k+1)]);

                                c_acc0 = _mm256_fmadd_pd(a0,  b0, c_acc0);
                                c_acc0 = _mm256_fmadd_pd(a0n, b1, c_acc0);
                                c_acc1 = _mm256_fmadd_pd(a1,  b0, c_acc1);
                                c_acc1 = _mm256_fmadd_pd(a1n, b1, c_acc1);
                            }
                            if (k < k_max){
                                const __m256d bv = _mm256_loadu_pd(b + k*ldb + j);
                                const __m256d a0 = _mm256_set1_pd(a[i*lda  + k]);
                                const __m256d a1 = _mm256_set1_pd(a[i1*lda + k]);
                                c_acc0 = _mm256_fmadd_pd(a0, bv, c_acc0);
                                c_acc1 = _mm256_fmadd_pd(a1, bv, c_acc1);
                            }

                            _mm256_storeu_pd(c + i*ldc  + j, c_acc0);
                            _mm256_storeu_pd(c + i1*ldc + j, c_acc1);
                        }

                        for(size_t j=j_vec_max; j<j_max; ++j){
                            double sum0 = c[i*ldc  + j];
                            double sum1 = c[i1*ldc + j];

                            size_t k = k0;
                            for(; k + 1 < k_max; k += 2){
                                sum0 += a[i*lda  + k]     * b[k*ldb     + j]
                                      + a[i*lda  + (k+1)] * b[(k+1)*ldb + j];
                                sum1 += a[i1*lda + k]     * b[k*ldb     + j]
                                      + a[i1*lda + (k+1)] * b[(k+1)*ldb + j];
                            }
                            if (k < k_max){
                                sum0 += a[i*lda  + k] * b[k*ldb + j];
                                sum1 += a[i1*lda + k] * b[k*ldb + j];
                            }

                            c[i*ldc  + j] = sum0;
                            c[i1*ldc + j] = sum1;
                        }
                    } else {
                        for(size_t j=j0; j<j_vec_max; j+=vec_size){
                            __m256d c_acc0 = _mm256_loadu_pd(c + i*ldc + j);

                            size_t k = k0;
                            for(; k + 1 < k_max; k += 2){
                                const __m256d b0 = _mm256_loadu_pd(b + k*ldb     + j);
                                const __m256d b1 = _mm256_loadu_pd(b + (k+1)*ldb + j);
                                const __m256d a0  = _mm256_set1_pd(a[i*lda + k]);
                                const __m256d a0n = _mm256_set1_pd(a[i*lda + (k+1)]);
                                c_acc0 = _mm256_fmadd_pd(a0,  b0, c_acc0);
                                c_acc0 = _mm256_fmadd_pd(a0n, b1, c_acc0);
                            }
                            if (k < k_max){
                                const __m256d bv = _mm256_loadu_pd(b + k*ldb + j);
                                const __m256d a0 = _mm256_set1_pd(a[i*lda + k]);
                                c_acc0 = _mm256_fmadd_pd(a0, bv, c_acc0);
                            }
                            _mm256_storeu_pd(c + i*ldc + j, c_acc0);
                        }

                        for(size_t j=j_vec_max; j<j_max; ++j){
                            double sum0 = c[i*ldc + j];
                            size_t k = k0;
                            for(; k + 1 < k_max; k += 2){
                                sum0 += a[i*lda + k]     * b[k*ldb     + j]
                                      + a[i*lda + (k+1)] * b[(k+1)*ldb + j];
                            }
                            if (k < k_max){
                                sum0 += a[i*lda + k] * b[k*ldb + j];
                            }
                            c[i*ldc + j] = sum0;
                        }
                    }
                } // i-loop
            } // j0
        } // i0
    } // k0

    return C;
}

// --------- mkl (dgemm) with delayed load ----------
Matrix multiply_mkl(const Matrix& A, const Matrix& B, double alpha=1.0, double beta=0.0){
    if (A.cols()!=B.rows()) throw std::runtime_error("dimension mismatch");

    auto dgemm = load_dgemm();
    if (!dgemm) {
        throw std::runtime_error(
            "BLAS library with cblas_dgemm not found. "
            "Install OpenBLAS/MKL or use LD_PRELOAD to provide it."
        );
    }

    const int M=(int)A.rows(), K=(int)A.cols(), N=(int)B.cols();
    Matrix C(M,N);
    dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          M, N, K, alpha,
          A.data(), K,
          B.data(), N,
          beta, C.data(), N);
    return C;
}

PYBIND11_MODULE(_matrix, m){
    m.doc()="HW4 matrix multiply with memory tracking allocator (delayed BLAS load)";

    py::class_<Matrix>(m,"Matrix",py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<size_t,size_t>())
        // validate.py compat attributes
        .def_property_readonly("rows",&Matrix::rows)
        .def_property_readonly("cols",&Matrix::cols)
        .def_property_readonly("nrow",&Matrix::rows)
        .def_property_readonly("ncol",&Matrix::cols)
        .def("fill",&Matrix::fill)
        .def("to_numpy",&Matrix::to_numpy)
        .def_static("from_numpy",&Matrix::from_numpy)
        .def_buffer(&Matrix::buffer_info)
        // Python index: A[i, j]
        .def("__getitem__", [](const Matrix& M, std::pair<size_t,size_t> ij){
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& M, std::pair<size_t,size_t> ij, double v){
            M(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& A, const Matrix& B){
            if (A.rows()!=B.rows() || A.cols()!=B.cols()) return false;
            const size_t n = A.rows()*A.cols();
            const double* ad = A.data();
            const double* bd = B.data();
            for (size_t i=0;i<n;++i){
                if (ad[i] != bd[i]) return false;
            }
            return true;
        });

    // multiply APIs
    m.def("multiply_naive",&multiply_naive);
    m.def("multiply_tile",&multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile")=128);
    m.def("multiple_tile",&multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile")=128); // alias
    m.def("multiply_mkl",&multiply_mkl, py::arg("A"), py::arg("B"), py::arg("alpha")=1.0, py::arg("beta")=0.0);

    // tracking APIs required by HW4
    m.def("bytes", [](){ return (py::int_)memtrack::bytes(); }, "Currently used bytes by Matrix buffers");
    m.def("allocated", [](){ return (py::int_)memtrack::allocated(); }, "Lifetime allocated bytes");
    m.def("deallocated", [](){ return (py::int_)memtrack::deallocated(); }, "Lifetime deallocated bytes");
}
