#include <pybind11/pybind11.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

double vector_angle(double x1, double y1, double x2, double y2)
{
    double dot = x1 * x2 + y1 * y2;
    double norm1 = std::sqrt(x1 * x1 + y1 * y1);
    double norm2 = std::sqrt(x2 * x2 + y2 * y2);

    if (norm1 == 0.0 || norm2 == 0.0)
        throw std::invalid_argument("Zero-length vector not allowed");

    double cos_theta = dot / (norm1 * norm2);
    
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    return std::acos(cos_theta);
}

PYBIND11_MODULE(vector_angle, m) { 
    m.def("vector_angle", &vector_angle, "Compute angle between 2D vectors"); 
}