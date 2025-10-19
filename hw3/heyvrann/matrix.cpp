#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#if defined(USE_MKL) || defined(HASMKL)
#include <mkl_cblas.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

struct Matrix {

    double* m_buffer = nullptr;

    Matrix(size_t nrow, size_t ncol): m_nrow(nrow), m_ncol(ncol) {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, const std::vector<double>& vec) {
            reset_buffer(nrow, ncol);
            *this = vec;
    }

    Matrix& operator=(const std::vector<double>& vec) {
        if (size() != vec.size()) 
            throw std::out_of_range("number of elements mismatch");

        size_t k = 0;
        for (size_t i = 0 ; i < m_nrow ; i++)
            for (size_t j = 0 ; j < m_ncol ; j++)
                (*this)(i, j) = vec[k++];

        return *this;
    }

    Matrix(const Matrix& other)
    : m_nflo(other.m_nflo) {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i = 0 ; i < m_nrow ; i++)
            for (size_t j = 0 ; j < m_ncol ; j++)
                (*this)(i, j) = other(i, j);
    }

    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;

        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
            reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i = 0 ; i < m_nrow ; i++)
            for (size_t j = 0 ; j < m_ncol ; j++)
                (*this)(i, j) = other(i, j);
        m_nflo = other.m_nflo;

        return *this;
    }

    Matrix(Matrix && other)
    : m_nflo(other.m_nflo) {
        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
    }

    Matrix& operator=(Matrix && other) {
        if (this == &other) return *this;

        reset_buffer(0, 0);
        std::swap(m_nrow, other.m_nrow);
        std::swap(m_ncol, other.m_ncol);
        std::swap(m_buffer, other.m_buffer);
        std::swap(m_nflo, other.m_nflo);

        return *this;
    }

    ~Matrix() { reset_buffer(0, 0); }

    double operator()(size_t row, size_t col) const { 
        return m_buffer[index(row, col)]; 
    }
    double& operator()(size_t row, size_t col) {
        return m_buffer[index(row, col)];
    }

    double operator[](size_t idx) const { return m_buffer[idx]; }
    double& operator[](size_t idx) { return m_buffer[idx]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }
    size_t size() const { return m_nrow * m_ncol; }

    size_t nflo() const { return m_nflo; }
    size_t& nflo() { return m_nflo; }

    Matrix transpose() const;

private:

    size_t m_nrow, m_ncol;
    size_t m_nflo = 0;

    size_t index(size_t row, size_t col) const { return row * m_ncol + col; }

    void reset_buffer(size_t nrow, size_t ncol) {
        if (m_buffer) delete[] m_buffer;

        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement]; }
        else { m_buffer = nullptr; }

        m_nrow = nrow; m_ncol = ncol;
    }
};

Matrix Matrix::transpose() const {
    Matrix ret(ncol(), nrow());
    for (size_t i = 0 ; i < ret.nrow() ; i++) 
        for (size_t j = 0 ; j < ret.ncol() ; j++)
            ret(i, j) = (*this)(j, i);
    return ret;
}

bool operator==(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.nrow() != mat2.nrow() || mat1.ncol() != mat2.ncol())
        return false;

    for (size_t i = 0 ; i < mat1.nrow() ; i++)
        for (size_t j = 0 ; j < mat1.ncol() ; j++) 
            if (mat1(i, j) != mat2(i, j))
                return false;


    return true;
}

bool operator!=(const Matrix& mat1, const Matrix& mat2) {
    return !(mat1 == mat2);
}

void validate_multiplication(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.ncol() != mat2.nrow()) {
        throw std::out_of_range("the number of first matrix column differs from that of second matrix row");
    }
}

size_t calculate_nflo(const Matrix& mat1, const Matrix& mat2) {
    return mat1.nrow() * mat1.ncol() * mat2.ncol();
}

Matrix multiply_naive(const Matrix& mat1, const Matrix& mat2) {
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t nrow = ret.nrow();
    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    for (size_t i = 0 ; i < nrow ; i++)
        for (size_t j = 0 ; j < ncol ; j++)
            ret(i, j) = 0;

    for (size_t i = 0 ; i < nrow1 ; i++) 
        for (size_t j = 0 ; j < ncol1 ; j++) 
            for (size_t k = 0 ; k < ncol2 ; k++)
                ret(i, k) += mat1(i, j) * mat2(j, k);

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}

Matrix multiply_tile(const Matrix& mat1, const Matrix& mat2, const size_t B) {
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol()), mat2_trans = mat2.transpose();

    const size_t nrow = ret.nrow();
    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    for (size_t i = 0 ; i < nrow ; i++)
        for (size_t j = 0 ; j < ncol ; j++)
            ret[i*ncol+j] = 0;

    for (size_t it = 0 ; it < nrow1 ; it += B) {
        const size_t n_i = std::min(it+B, nrow1); 
        for (size_t jt = 0 ; jt < ncol2 ; jt += B) {
            const size_t n_j = std::min(jt+B, ncol2);
            for (size_t kt = 0  ; kt < ncol1 ; kt += B) {
                const size_t n_k = std::min(kt+B, ncol1);
                for (size_t i = it ; i < n_i ; i++) {
                    for (size_t j = jt ; j < n_j ; j++) {
                        for (size_t k = kt ; k < n_k ; k++) {
                            ret.m_buffer[i*ncol+j] += mat1.m_buffer[i*ncol1+k] * mat2_trans.m_buffer[j*ncol1+k];
                        }
                    }
                }
            }
        }
    }

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}

Matrix multiply_mkl(const Matrix& mat1, const Matrix& mat2) {
#if !defined(HASMKL) || defined(NOMKL)
    // run with VECLIB_MAXIMUM_THREAD=1
#else // HASMKL NOMKL
    mkl_set_num_threads(1);
#endif //HASMKL NOMKL

    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nrow1, ncol2, ncol1, 1.0, 
                mat1.m_buffer, ncol1,
                mat2.m_buffer, ncol2,
                0.0, ret.m_buffer, ncol);

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    m.doc() = "Python bindings for Matrix class and multiplication functions";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", [](const Matrix& mat) { return mat.nrow(); })
        .def_property_readonly("ncol", [](const Matrix& mat) { return mat.ncol(); })
        .def("__getitem__", [](const Matrix& mat, std::pair<size_t, size_t> idx) {
            return mat(idx.first, idx.second);
        }) 
        .def("__setitem__", [](Matrix& mat, std::pair<size_t, size_t> idx, double val) {
            mat(idx.first, idx.second) = val;
        })
        .def("__eq__", [](const Matrix& mat1, const Matrix& mat2) { return mat1 == mat2; })
        .def("transpose", &Matrix::transpose)
        .def_property_readonly("nflo", [](const Matrix& mat) { return mat.nflo(); });

    m.def("multiply_naive", &multiply_naive, "Naive matrix multiplication");
    m.def("multiply_tile", &multiply_tile, "Tiled matrix multiplication",
          py::arg("mat1"), py::arg("mat2"), py::arg("B") = 64);
    m.def("multiply_mkl", &multiply_mkl, "Matrix multiplication using MKL");
}