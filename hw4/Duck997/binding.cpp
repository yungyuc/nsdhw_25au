#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("rows", &Matrix::rows)
        .def("cols", &Matrix::cols)
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols)
        .def("at", (double& (Matrix::*)(size_t,size_t)) &Matrix::at,
             py::return_value_policy::reference_internal)
        .def("get", &Matrix::get)
        .def("set", &Matrix::set)
        .def("__getitem__", [](Matrix const& M, std::pair<size_t,size_t> ij){
            return M.get(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix & M, std::pair<size_t,size_t> ij, double v){
            M.set(ij.first, ij.second, v);
        })
        .def("__eq__", [](Matrix const& A, Matrix const& B){
            if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
            for (size_t i=0;i<A.rows();++i){
                for (size_t j=0;j<A.cols();++j){
                    if (A.at(i,j) != B.at(i,j)) return false;
                }
            }
            return true;
        });

    m.def("multiply_naive", &multiply_naive);
    m.def("multiple_tile", &multiple_tile, py::arg("A"), py::arg("B"), py::arg("tile") = 32);
    m.def("multiply_tile", &multiple_tile, py::arg("A"), py::arg("B"), py::arg("tsize") = 32);
    m.def("multiply_mkl", &multiply_mkl);
    m.def("blas_available", &blas_available);
    m.def("bytes", &bytes);
    m.def("allocated", &allocated);
    m.def("deallocated", &deallocated);
}


