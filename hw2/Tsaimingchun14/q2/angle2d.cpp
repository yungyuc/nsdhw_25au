#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <array>
#include <stdexcept>

double angle2d(const std::array<double, 2>& v1, const std::array<double, 2>& v2)
{
    double dot = v1[0]*v2[0] + v1[1]*v2[1];
    double norm1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1]);
    double norm2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1]);

    if (norm1 == 0.0 || norm2 == 0.0)
        throw std::invalid_argument("Zero-length vector");

    double cos_theta = dot / (norm1 * norm2);
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    return std::acos(cos_theta);
}

PYBIND11_MODULE(_vector, m) {
    m.doc() = "2D vector angle calculation";
    m.def("angle2d", &angle2d,
          "Calculate angle between two 2D vectors",
          pybind11::arg("v1"), pybind11::arg("v2"));
}

