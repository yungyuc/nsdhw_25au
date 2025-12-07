#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def_property_readonly("array", &Matrix::array)
        .def("__eq__", [](Matrix const & self, Matrix const & other) { return self == other; })
        .def("__getitem__", [](Matrix const & self, std::tuple<size_t, size_t> idx) {
            return self(std::get<0>(idx), std::get<1>(idx));
        })
        .def("__setitem__", [](Matrix & self, std::tuple<size_t, size_t> idx, double value) {
            return self(std::get<0>(idx), std::get<1>(idx)) = value;
        });

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("multiply_tile", &multiply_tile);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
