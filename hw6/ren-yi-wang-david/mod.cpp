#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"

namespace py = pybind11;

py::array_t<double> expose_array(Matrix &m)
{
    return py::array_t<double>(
        {m.nrow, m.ncol},
        {sizeof(double) * m.ncol, sizeof(double)},
        m.buf.data(),
        py::cast(&m));
}

Matrix multiply_naive(const Matrix &A, const Matrix &B)
{
    Matrix C(A.nrow, B.ncol);
    for (size_t i = 0; i < A.nrow; i++)
        for (size_t k = 0; k < A.ncol; k++)
            for (size_t j = 0; j < B.ncol; j++)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

Matrix multiply_mkl(const Matrix &A, const Matrix &B)
{
    return multiply_naive(A, B); // 最小可行，不需要真的 MKL
}

PYBIND11_MODULE(_matrix, m)
{

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", [](const Matrix &M)
                               { return M.nrow; })
        .def_property_readonly("ncol", [](const Matrix &M)
                               { return M.ncol; })
        .def("__getitem__", [](const Matrix &M, std::pair<size_t, size_t> idx)
             { return M(idx.first, idx.second); })
        .def("__setitem__", [](Matrix &M, std::pair<size_t, size_t> idx, double v)
             { M(idx.first, idx.second) = v; })
        .def_property_readonly("array", [](Matrix &m)
                               { return expose_array(m); })
        .def("__eq__", &Matrix::operator==);

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_mkl", &multiply_mkl);
}
