#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cassert>
#include <mutex>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

// ===================== Global Memory Tracker =====================
static std::size_t g_bytes = 0;
static std::size_t g_allocated = 0;
static std::size_t g_deallocated = 0;
static std::mutex g_mutex;

inline void add_allocation(std::size_t n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_bytes += n;
    g_allocated += n;
}
inline void add_deallocation(std::size_t n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_bytes >= n)
        g_bytes -= n;
    g_deallocated += n;
}

// ===================== Custom Allocator =====================
template <typename T>
struct CustomAllocator {
    using value_type = T;
    CustomAllocator() noexcept {}
    template <typename U> CustomAllocator(const CustomAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        add_allocation(bytes);
        return static_cast<T*>(::operator new(bytes));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::size_t bytes = n * sizeof(T);
        add_deallocation(bytes);
        ::operator delete(p);
    }
};

// ===================== Matrix Class =====================
class Matrix {
public:
    int nrow, ncol;
    std::vector<double, CustomAllocator<double>> data;

    Matrix(int r, int c) : nrow(r), ncol(c), data(r * c, 0.0) {}

    double& operator()(int r, int c) { return data[r * ncol + c]; }
    const double& operator()(int r, int c) const { return data[r * ncol + c]; }

    Matrix naive_mul(const Matrix& B) const {
        assert(ncol == B.nrow);
        Matrix C(nrow, B.ncol);
        for (int i = 0; i < nrow; ++i)
            for (int k = 0; k < ncol; ++k) {
                double val = (*this)(i, k);
                for (int j = 0; j < B.ncol; ++j)
                    C(i, j) += val * B(k, j);
            }
        return C;
    }

    Matrix mkl_mul(const Matrix& B) const {
        assert(ncol == B.nrow);
        Matrix C(nrow, B.ncol);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nrow, B.ncol, ncol,
                    1.0, data.data(), ncol,
                    B.data.data(), B.ncol,
                    0.0, C.data.data(), C.ncol);
        return C;
    }

    bool operator==(const Matrix& other) const {
        if (nrow != other.nrow || ncol != other.ncol) return false;
        for (int i = 0; i < nrow * ncol; ++i)
            if (data[i] != other.data[i]) return false;
        return true;
    }
};

// ===================== Python Binding =====================
namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int, int>())
        .def_readonly("nrow", &Matrix::nrow)
        .def_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](const Matrix& self, std::pair<int, int> idx) {
            return self(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& self, std::pair<int, int> idx, double v) {
            self(idx.first, idx.second) = v;
        })
        .def("__eq__", &Matrix::operator==);

    // 對應 validate.py 測試
    m.def("multiply_naive", [](const Matrix& A, const Matrix& B) { return A.naive_mul(B); });
    m.def("multiply_mkl",   [](const Matrix& A, const Matrix& B) { return A.mkl_mul(B); });

    // 記憶體統計
    m.def("bytes", []() { return g_bytes; });
    m.def("allocated", []() { return g_allocated; });
    m.def("deallocated", []() { return g_deallocated; });
}
