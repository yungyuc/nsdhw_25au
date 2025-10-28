#include "mul_set.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t,size_t>())
        .def("nrow", &Matrix::nrow)
        .def("ncol", &Matrix::ncol)
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](const Matrix& M, const std::pair<size_t,size_t>& ij){
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& M, const std::pair<size_t,size_t>& ij, double v){
            M(ij.first, ij.second) = v;
        });

    m.def("populate",       &populate);
    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile",  &multiply_tile);
    m.def("multiply_mkl",   &multiply_mkl);
}