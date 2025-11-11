#include <mkl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <atomic>
#include <algorithm>

// ======================= Memory Tracking =========================
static std::atomic<size_t> g_bytes_used{0};
static std::atomic<size_t> g_allocated{0};
static std::atomic<size_t> g_deallocated{0};

// ======================= Custom Allocator =========================
template <typename T>
struct CustomAllocator {
    using value_type = T;

    CustomAllocator() = default;
    template <class U>
    constexpr CustomAllocator(const CustomAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        g_bytes_used += bytes;
        g_allocated += bytes;
        return static_cast<T*>(::operator new(bytes));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        std::size_t bytes = n * sizeof(T);
        g_bytes_used -= bytes;
        g_deallocated += bytes;
        ::operator delete(p);
    }
};

// ============================ Matrix ==============================
class Matrix {
public:
    size_t nrow;
    size_t ncol;
    std::vector<double, CustomAllocator<double>> m_buffer;

    Matrix(size_t n_row, size_t n_col)
        : nrow(n_row), ncol(n_col), m_buffer(n_row * n_col, 0.0) {}

    double operator()(size_t row, size_t col) const {
        return m_buffer[row * ncol + col];
    }
    double& operator()(size_t row, size_t col) {
        return m_buffer[row * ncol + col];
    }

    void load_from_python(pybind11::array_t<double> input) {
        pybind11::buffer_info buf = input.request();
        std::memcpy(m_buffer.data(), buf.ptr, nrow * ncol * sizeof(double));
    }

    void load_buffer(const double* input) {
        std::memcpy(m_buffer.data(), input, nrow * ncol * sizeof(double));
    }

    friend bool operator==(const Matrix& A, const Matrix& B) {
        if (A.nrow != B.nrow || A.ncol != B.ncol) return false;
        return std::equal(A.m_buffer.begin(), A.m_buffer.end(), B.m_buffer.begin());
    }

    double* get_data() const { return const_cast<double*>(m_buffer.data()); }

    std::string tostring() const {
        std::stringstream ss;
        ss << "[";
        for (size_t r = 0; r < nrow; ++r) {
            if (r > 0) ss << " ";
            ss << "[";
            for (size_t c = 0; c < ncol; ++c)
                ss << (*this)(r, c) << " ";
            ss << "]";
            if (r < nrow - 1) ss << "\n";
        }
        ss << "]\n";
        return ss.str();
    }
};

// ======================= Multiplication ===========================
Matrix multiply_naive(Matrix& A, Matrix& B) {
    if (A.ncol != B.nrow)
        throw pybind11::value_error("Matrices' shape are not matched.");

    Matrix C(A.nrow, B.ncol);
    for (size_t i = 0; i < A.nrow; ++i)
        for (size_t j = 0; j < B.ncol; ++j)
            for (size_t k = 0; k < A.ncol; ++k)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

Matrix multiply_tile(Matrix& A, Matrix& B, size_t tsize) {
    if (A.ncol != B.nrow)
        throw pybind11::value_error("Matrices' shape are not matched.");

    Matrix C(A.nrow, B.ncol);
    for (size_t i0 = 0; i0 < A.nrow; i0 += tsize)
        for (size_t j0 = 0; j0 < B.ncol; j0 += tsize)
            for (size_t k = 0; k < A.ncol; ++k)
                for (size_t i = i0; i < std::min(i0 + tsize, A.nrow); ++i)
                    for (size_t j = j0; j < std::min(j0 + tsize, B.ncol); ++j)
                        C(i, j) += A(i, k) * B(k, j);
    return C;
}

Matrix multiply_mkl(Matrix& A, Matrix& B) {
    if (A.ncol != B.nrow)
        throw pybind11::value_error("Matrices' shape are not matched.");

    Matrix res(A.nrow, B.ncol);

    int m = static_cast<int>(A.nrow);
    int n = static_cast<int>(B.ncol);
    int k = static_cast<int>(A.ncol);
    double alpha = 1.0, beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                A.get_data(), k,
                B.get_data(), n,
                beta, res.get_data(), n);

    return res;
}

// ======================== Python Bindings =========================
PYBIND11_MODULE(_matrix, m) {
    m.doc() = "_matrix with custom allocator and tracking";

    pybind11::class_<Matrix>(m, "Matrix")
        .def(pybind11::init<size_t, size_t>())
        .def("__getitem__", [](Matrix& mat, std::pair<size_t, size_t> idx) {
            return mat(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix& mat, std::pair<size_t, size_t> idx, double val) {
            mat(idx.first, idx.second) = val;
        })
        .def("__str__", &Matrix::tostring)
        .def("load", &Matrix::load_from_python)
        .def(pybind11::self == pybind11::self)
        .def_readonly("nrow", &Matrix::nrow)
        .def_readonly("ncol", &Matrix::ncol);

    // Matrix multiplication bindings
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile);
    m.def("multiply_mkl", &multiply_mkl);

    // Memory tracking functions
    m.def("bytes", []() { return g_bytes_used.load(); });
    m.def("allocated", []() { return g_allocated.load(); });
    m.def("deallocated", []() { return g_deallocated.load(); });
}
