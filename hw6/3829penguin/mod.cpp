#include "matrix.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())

        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)

        .def("__getitem__", [](const Matrix &M, std::pair<size_t,size_t> idx){
            return M(idx.first, idx.second);
        })
        .def("__setitem__", [](Matrix &M, std::pair<size_t,size_t> idx, double v){
            M(idx.first, idx.second) = v;
        })

        .def_property_readonly("array", [](Matrix &M){
            return py::array_t<double>(
                {M.nrow(), M.ncol()},
                {sizeof(double) * M.ncol(), sizeof(double)},
                M.get_data(),
                py::cast(&M)
            );
        })

        .def("__eq__", [](const Matrix &A, const Matrix &B){
            return A == B;
        })
    ;

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_mkl", &multiply_mkl);
}


