#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstring>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <dlfcn.h>
#include <immintrin.h>

namespace py = pybind11;

namespace tracker {
    static std::atomic<size_t> used_bytes{0};
    static std::atomic<size_t> alloc_bytes{0};
    static std::atomic<size_t> free_bytes{0};

    inline void on_alloc(size_t b) {
        alloc_bytes += b;
        used_bytes += b;
    }
    inline void on_free(size_t b) {
        free_bytes += b;
        used_bytes -= b;
    }

    inline size_t current() { return used_bytes.load(); }
    inline size_t total_alloc() { return alloc_bytes.load(); }
    inline size_t total_free() { return free_bytes.load(); }
}

template <typename T>
struct TrackedAlloc {
    using value_type = T;

    TrackedAlloc() noexcept {}
    template <typename U> TrackedAlloc(const TrackedAlloc<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        tracker::on_alloc(bytes);
        return static_cast<T*>(::operator new(bytes));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::size_t bytes = n * sizeof(T);
        tracker::on_free(bytes);
        ::operator delete(p);
    }
};

template <typename T, typename U>
bool operator==(const TrackedAlloc<T>&, const TrackedAlloc<U>&) { return true; }
template <typename T, typename U>
bool operator!=(const TrackedAlloc<T>&, const TrackedAlloc<U>&) { return false; }

using dgemm_t = void(*)(int, int, int, int, int, int,
                        double, const double*, int,
                        const double*, int, double,
                        double*, int);

static dgemm_t load_blas() {
    static dgemm_t fn = nullptr;
    static bool tried = false;
    if (tried) return fn;
    tried = true;

    const char* libs[] = {
        "libopenblas.so", "libopenblas.so.0", "libmkl_rt.so", "libblas.so"
    };
    for (auto so : libs) {
        void* handle = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) continue;
        fn = (dgemm_t)dlsym(handle, "cblas_dgemm");
        if (fn) return fn;
        dlclose(handle);
    }
    return nullptr;
}

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t r, size_t c, double val = 0.0)
        : rows_(r), cols_(c), data_(r * c, val) {}

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    double& operator()(size_t i, size_t j) noexcept { return data_[i * cols_ + j]; }
    double  operator()(size_t i, size_t j) const noexcept { return data_[i * cols_ + j]; }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    py::array_t<double> to_numpy() const {
        return py::array_t<double>(
            {(py::ssize_t)rows_, (py::ssize_t)cols_},
            {(py::ssize_t)(cols_ * sizeof(double)), (py::ssize_t)sizeof(double)},
            data_.data()
        );
    }

    static Matrix from_numpy(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2) throw std::runtime_error("expected 2D array");
        size_t r = arr.shape(0), c = arr.shape(1);
        Matrix m(r, c);
        std::memcpy(m.data(), arr.data(), sizeof(double) * r * c);
        return m;
    }

private:
    size_t rows_, cols_;
    std::vector<double, TrackedAlloc<double>> data_;
};


Matrix mul_basic(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix C(M, N);
    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k) {
            double a = A(i, k);
            for (size_t j = 0; j < N; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

Matrix mul_blocked(const Matrix& A, const Matrix& B, int block = 64) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix C(M, N);
    for (size_t ii = 0; ii < M; ii += block)
        for (size_t jj = 0; jj < N; jj += block)
            for (size_t kk = 0; kk < K; kk += block) {
                size_t i_end = std::min(ii + (size_t)block, M);
                size_t j_end = std::min(jj + (size_t)block, N);
                size_t k_end = std::min(kk + (size_t)block, K);
                for (size_t i = ii; i < i_end; ++i)
                    for (size_t k = kk; k < k_end; ++k) {
                        double a = A(i, k);
                        for (size_t j = jj; j < j_end; ++j)
                            C(i, j) += a * B(k, j);
                    }
            }
    return C;
}

Matrix mul_blas(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
    auto dgemm = load_blas();
    if (!dgemm) throw std::runtime_error("No cblas_dgemm found on system.");

    const int M = (int)A.rows(), K = (int)A.cols(), N = (int)B.cols();
    Matrix C(M, N);
    dgemm(101, 111, 111, M, N, K, 1.0,
          A.data(), K, B.data(), N, 0.0, C.data(), N);
    return C;
}

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<>())
        .def(py::init<size_t, size_t, double>(), py::arg("rows"), py::arg("cols"), py::arg("init")=0.0)
        .def("fill", &Matrix::fill)
        .def("to_numpy", &Matrix::to_numpy)
        .def_static("from_numpy", &Matrix::from_numpy)
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def("__getitem__", [](const Matrix& M, std::pair<size_t, size_t> ij){ return M(ij.first, ij.second); })
        .def("__setitem__", [](Matrix& M, std::pair<size_t, size_t> ij, double v){ M(ij.first, ij.second)=v; });

    // functions
    m.def("multiply_naive", &mul_basic);
    m.def("multiply_tile", &mul_blocked, py::arg("A"), py::arg("B"), py::arg("block")=64);
    m.def("multiply_mkl", &mul_blas);

    // memory stats
    m.def("bytes", [](){ return tracker::current(); });
    m.def("allocated", [](){ return tracker::total_alloc(); });
    m.def("deallocated", [](){ return tracker::total_free(); });
}
