#include <stdexcept>
#include <vector>
#include <cstddef>
#include <algorithm> // For std::min
#include <atomic>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(USE_MKL)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

// Global atomic counters for memory tracking
static std::atomic<size_t> current_bytes_used(0);
static std::atomic<size_t> total_bytes_allocated(0);
static std::atomic<size_t> total_bytes_deallocated(0);

template <class T>
struct CustomAllocator
{
    using value_type = T;

    CustomAllocator() = default;

    template <class U>
    constexpr CustomAllocator(const CustomAllocator<U> &) noexcept {}

    T *allocate(size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }

        const size_t bytes = n * sizeof(T);
        T *p = static_cast<T *>(std::malloc(bytes));
        if (p)
        {
            current_bytes_used += bytes;
            total_bytes_allocated += bytes;
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T *p, size_t n) noexcept
    {
        const size_t bytes = n * sizeof(T);
        std::free(p);
        current_bytes_used -= bytes;
        total_bytes_deallocated += bytes;
    }
};

template <class T, class U>
bool operator==(const CustomAllocator<T> &, const CustomAllocator<U> &)
{
    return true;
}

template <class T, class U>
bool operator!=(const CustomAllocator<T> &, const CustomAllocator<U> &)
{
    return false;
}

// Functions to be exposed to Python
size_t get_bytes()
{
    return current_bytes_used.load();
}

size_t get_allocated()
{
    return total_bytes_allocated.load();
}

size_t get_deallocated()
{
    return total_bytes_deallocated.load();
}

class Matrix
{
public:
    Matrix(size_t rows, size_t cols);

    size_t rows() const;
    size_t cols() const;

    double &operator()(size_t r, size_t c);
    const double &operator()(size_t r, size_t c) const;

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;

    Matrix transpose() const;

    std::vector<double, CustomAllocator<double>> &data();
    const std::vector<double, CustomAllocator<double>> &data() const;

private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double, CustomAllocator<double>> m_data;
};

Matrix::Matrix(size_t rows, size_t cols)
    : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0) {}
size_t Matrix::rows() const { return m_rows; }
size_t Matrix::cols() const { return m_cols; }
double &Matrix::operator()(size_t r, size_t c) { return m_data[r * m_cols + c]; }
const double &Matrix::operator()(size_t r, size_t c) const { return m_data[r * m_cols + c]; }
bool Matrix::operator==(const Matrix &other) const
{
    if (m_rows != other.m_rows || m_cols != other.m_cols)
        return false;
    return m_data == other.m_data;
}
bool Matrix::operator!=(const Matrix &other) const { return !(*this == other); }
std::vector<double, CustomAllocator<double>> &Matrix::data() { return m_data; }
const std::vector<double, CustomAllocator<double>> &Matrix::data() const { return m_data; }

Matrix Matrix::transpose() const
{
    Matrix result(m_cols, m_rows);
    for (size_t i = 0; i < m_rows; ++i)
    {
        for (size_t j = 0; j < m_cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Naive implementation remains the same
Matrix multiply_naive(const Matrix &a, const Matrix &b)
{
    if (a.cols() != b.rows())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }
    Matrix result(a.rows(), b.cols());
    for (size_t i = 0; i < a.rows(); ++i)
    {
        for (size_t j = 0; j < b.cols(); ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < a.cols(); ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Tiling implementation using the transpose strategy
Matrix multiply_tile(const Matrix &a, const Matrix &b, size_t tile_size)
{
    if (a.cols() != b.rows())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible.");
    }

    // Step 1: Create the transposed version of B
    Matrix b_t = b.transpose();
    Matrix c(a.rows(), b.cols());
    size_t M = a.rows(), N = b.cols(), K = a.cols();
    const auto &A = a.data();
    const auto &B_t = b_t.data();
    auto &C = c.data();

    for (size_t i0 = 0; i0 < M; i0 += tile_size)
    {
        size_t i_max = std::min(i0 + tile_size, M);
        for (size_t j0 = 0; j0 < N; j0 += tile_size)
        {
            size_t j_max = std::min(j0 + tile_size, N);
            for (size_t k0 = 0; k0 < K; k0 += tile_size)
            {
                size_t k_max = std::min(k0 + tile_size, K);

                // Step 2: Use simple i,j,k loop order with the transposed matrix
                for (size_t i = i0; i < i_max; ++i)
                {
                    for (size_t j = j0; j < j_max; ++j)
                    {
                        double sum = C[i * N + j];
                        for (size_t k = k0; k < k_max; ++k)
                        {
                            // Access b_t(j, k) instead of b(k, j)
                            sum += A[i * K + k] * B_t[j * K + k];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }

    return c;
}

// MKL implementation
Matrix multiply_mkl(const Matrix &a, const Matrix &b)
{
    if (a.cols() != b.rows())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }
    Matrix result(a.rows(), b.cols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a.rows(), b.cols(), a.cols(), 1.0,
                a.data().data(), a.cols(),
                b.data().data(), b.cols(), 0.0,
                result.data().data(), b.cols());
    return result;
}

namespace py = pybind11;
PYBIND11_MODULE(_matrix, m)
{
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("__eq__", &Matrix::operator==)
        .def("__ne__", &Matrix::operator!=)
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols)
        .def("__getitem__", [](const Matrix &m, std::pair<size_t, size_t> i)
             { return m(i.first, i.second); })
        .def("__setitem__", [](Matrix &m, std::pair<size_t, size_t> i, double v)
             { m(i.first, i.second) = v; });

    m.def("multiply_naive", &multiply_naive, "Naive matrix multiplication");
    m.def("multiply_tile", &multiply_tile, "Tiled matrix multiplication");
    m.def("multiply_mkl", &multiply_mkl, "MKL-based matrix multiplication");

    m.def("bytes", &get_bytes, "Return the current number of bytes used by all Matrix instances.");
    m.def("allocated", &get_allocated, "Return the total number of bytes allocated by all Matrix instances.");
    m.def("deallocated", &get_deallocated, "Return the total number of bytes deallocated by all Matrix instances.");
}