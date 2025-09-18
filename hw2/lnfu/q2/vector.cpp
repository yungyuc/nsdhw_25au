#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

double get_angle(double x1, double y1, double x2, double y2)
{
    double magnitude1 = std::sqrt(x1 * x1 + y1 * y1);
    double magnitude2 = std::sqrt(x2 * x2 + y2 * y2);

    if (magnitude1 == 0.0 || magnitude2 == 0.0)
    {
        throw std::invalid_argument("One or both vectors are zero vectors.");
    }

    double cos_theta = (x1 * x2 + y1 * y2) / (magnitude1 * magnitude2);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    return std::acos(cos_theta);
}

PYBIND11_MODULE(_vector, m)
{
    m.doc() = "Module for angle calculation between two 2D vectors";
    m.def("get_angle", &get_angle,
          "Calculates the angle (in radians) between two vectors in the 2-dimensional Cartesian coordinate system",
          pybind11::arg("x1"), pybind11::arg("y1"),
          pybind11::arg("x2"), pybind11::arg("y2"));
}