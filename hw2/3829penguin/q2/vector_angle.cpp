#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace py = pybind11;

double angleBetweenVectors(double x1, double y1, double x2, double y2) {
    double dot = x1 * x2 + y1 * y2;
    double len1 = std::sqrt(x1 * x1 + y1 * y1);
    double len2 = std::sqrt(x2 * x2 + y2 * y2);
    if (len1 == 0.0 || len2 == 0.0) {
        throw std::invalid_argument("vector length is zero");
    }
    double cosTheta = dot / (len1 * len2);
    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;
    return std::acos(cosTheta);
}


PYBIND11_MODULE(_vector, m) {
    m.doc() = "Module to compute angle between 2D vectors";
    m.def("angle_between_vectors", &angleBetweenVectors, "Compute angle between two 2D vectors");
}
