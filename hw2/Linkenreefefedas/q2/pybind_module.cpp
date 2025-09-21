#include <pybind11/pybind11.h>
#include "angle.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vecgeom, m) {
    m.doc() = "2D vector angle utilities";
    m.def("angle_between",
          [](double ax, double ay, double bx, double by) {
              return angle_between(ax, ay, bx, by);
          },
          py::arg("ax"), py::arg("ay"), py::arg("bx"), py::arg("by"),
          "Return the angle (radians) between 2D vectors (ax,ay) and (bx,by).");
}
