#include <pybind11/pybind11.h>
#include "vector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_vector, m) {
    m.def("angle_between_vectors", &vector_angle_rad,
          "Compute angle between two 2D vectors (radians)");
}
