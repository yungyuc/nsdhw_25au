#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Mat>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>())
        .def_property_readonly("nrow", &Mat::nrow)
        .def_property_readonly("ncol", &Mat::ncol)
        .def_property_readonly("array", &Mat::array)
        .def("__getitem__", [](const Mat& M, std::pair<std::size_t, std::size_t> ij) {
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Mat& M, std::pair<std::size_t, std::size_t> ij, double v) {
            M(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Mat& a, const Mat& b) { return a == b; });

    m.def("multiply_naive", &mul_naive);
    m.def("multiply_mkl", &mul_mkl);
}
