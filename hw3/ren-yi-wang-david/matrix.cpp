#include "StopWatch.hpp"


#ifdef HASMKL
#include <mkl_service.h>  
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#elif defined(__MACH__)
#include <clapack.h>
#include <Accelerate/Accelerate.h>
#endif

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

struct Matrix {
public:
    Matrix(size_t nrow, size_t ncol)
        : m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const &vec)
        : m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix &operator=(std::vector<double> const &vec) {
        if (size() != vec.size())
            throw std::out_of_range("number of elements mismatch");

        size_t k = 0;
        for (size_t i = 0; i < m_nrow; ++i)
            for (size_t j = 0; j < m_ncol; ++j)
                (*this)(i, j) = vec[k++];
        return *this;
    }

    double operator()(size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double &operator()(size_t row, size_t col) { return m_buffer[index(row, col)]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    size_t size() const { return m_nrow * m_ncol; }

    double elapsed() const { return m_elapsed; }
    double gflops() const { return m_nflo / m_elapsed / 1e9; }

    double *data() { return m_buffer; }

    void reset_buffer(size_t nrow, size_t ncol) {
        if (m_buffer) delete[] m_buffer;
        size_t n = nrow * ncol;
        m_buffer = n ? new double[n] : nullptr;
        m_nrow = nrow; m_ncol = ncol;
    }

    size_t index(size_t row, size_t col) const { return row * m_ncol + col; }

    size_t m_nrow = 0, m_ncol = 0;
    double *m_buffer = nullptr;
    double m_elapsed = 0;
    size_t m_nflo = 0;
};

// ---- Utility ----
void validate_multiplication(Matrix const &A, Matrix const &B) {
    if (A.ncol() != B.nrow())
        throw std::out_of_range("shape mismatch");
}

size_t calc_nflo(Matrix const &A, Matrix const &B) {
    return A.nrow() * A.ncol() * B.ncol();
}

// ---- MKL ----
Matrix multiply_mkl(Matrix const &A, Matrix const &B) {
#if defined(HASMKL)
    mkl_set_num_threads(1);
#endif
    validate_multiplication(A, B);
    Matrix C(A.nrow(), B.ncol());
    StopWatch sw;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A.nrow(), B.ncol(), A.ncol(),
                1.0, A.m_buffer, A.ncol(),
                B.m_buffer, B.ncol(),
                0.0, C.m_buffer, C.ncol());

    C.m_elapsed = sw.lap();
    C.m_nflo = calc_nflo(A, B);
    return C;
}

// ---- Naive ----
Matrix multiply_naive(Matrix const &A, Matrix const &B) {
    validate_multiplication(A, B);
    Matrix C(A.nrow(), B.ncol());
    StopWatch sw;

    for (size_t i = 0; i < A.nrow(); ++i)
        for (size_t k = 0; k < B.ncol(); ++k) {
            double v = 0;
            for (size_t j = 0; j < A.ncol(); ++j)
                v += A(i, j) * B(j, k);
            C(i, k) = v;
        }

    C.m_elapsed = sw.lap();
    C.m_nflo = calc_nflo(A, B);
    return C;
}

// ---- Tiled ----
Matrix multiply_tile(Matrix const &A, Matrix const &B, int block = 64) {
    validate_multiplication(A, B);
    Matrix C(A.nrow(), B.ncol());
    StopWatch sw;

    for (int ii = 0; ii < (int)A.nrow(); ii += block)
        for (int jj = 0; jj < (int)B.ncol(); jj += block)
            for (int kk = 0; kk < (int)A.ncol(); kk += block)
                for (int i = ii; i < std::min(ii + block, (int)A.nrow()); ++i)
                    for (int j = jj; j < std::min(jj + block, (int)B.ncol()); ++j) {
                        double sum = C(i, j);
                        for (int k = kk; k < std::min(kk + block, (int)A.ncol()); ++k)
                            sum += A(i, k) * B(k, j);
                        C(i, j) = sum;
                    }

    C.m_elapsed = sw.lap();
    C.m_nflo = calc_nflo(A, B);
    return C;
}



PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](const Matrix &self, std::pair<size_t, size_t> idx) {
            return self(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix &self, std::pair<size_t, size_t> idx, double value) {
            self(idx.first, idx.second) = value;
        })
        // ✅ 新增這三個讓 Python 可以取出計時與運算資訊
        .def("elapsed", &Matrix::elapsed)
        .def("gflops", &Matrix::gflops)
        .def_property_readonly("size", &Matrix::size);

    // 綁定三種乘法函數
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile", &multiply_tile, py::arg("A"), py::arg("B"), py::arg("block") = 64);
    m.def("multiply_mkl", &multiply_mkl);
}
