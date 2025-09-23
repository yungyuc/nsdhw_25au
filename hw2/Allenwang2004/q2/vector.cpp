#include <cmath>
#include <pybind11/pybind11.h>

double angle_between(double x1, double y1, double x2, double y2) {
    double dot = x1*x2 + y1*y2;
    double mag1 = std::sqrt(x1*x1 + y1*y1);
    double mag2 = std::sqrt(x2*x2 + y2*y2);
    return std::acos(dot / (mag1 * mag2));
}

PYBIND11_MODULE(_vector, m) {
    m.def("angle_between", &angle_between, "Calculate the angle between two vectors");
}