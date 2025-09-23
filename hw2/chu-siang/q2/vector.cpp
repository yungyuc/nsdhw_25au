#include <pybind11/pybind11.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

double angle(double x1, double y1, double x2, double y2) {
    const double n1 = std::hypot(x1, y1);
    const double n2 = std::hypot(x2, y2);
    if (n1 == 0.0 || n2 == 0.0) {
        throw std::invalid_argument("zero-length vector");
    }
    const double dot = x1 * x2 + y1 * y2;
    double c = dot / (n1 * n2);
    if (c > 1.0)  c = 1.0;
    if (c < -1.0) c = -1.0;
    return std::acos(c);
}

PYBIND11_MODULE(_vector, m) {
    m.doc() = "2D vector utilities";
    m.def("angle", &angle, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
          "Compute the angle (radians) between two 2D vectors.");
}

