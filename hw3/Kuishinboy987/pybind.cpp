#include "mul_set.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t,size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](const Matrix& M, const std::pair<size_t,size_t>& ij){
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& M, const std::pair<size_t,size_t>& ij, double v){
            M(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& A, const Matrix& B){
            if (A.nrow() != B.nrow() || A.ncol() != B.ncol()) return false;
            for (size_t i = 0; i < A.nrow(); ++i)
                for (size_t j = 0; j < A.ncol(); ++j)
                    if (A(i,j) != B(i,j)) return false;
            return true;
        });

    m.def("populate",       &populate);
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile",  &multiply_tile);
    m.def("multiply_mkl",   &multiply_mkl);
}
