#include <pybind11/pybind11.h>
#include "angle.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_vector, m) {
    m.doc() = "2D vector helpers: angle_between((x1,y1), (x2,y2)) in radians";

    m.def("angle_between",
          [](double x1, double y1, double x2, double y2) {
              return angle_between(x1, y1, x2, y2);
          },
          py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
          "Return angle (radians) between two 2D vectors (x1,y1) and (x2,y2).");

    m.def("angle_between_xy",
          [](py::tuple v1, py::tuple v2) {
              if (py::len(v1) != 2 || py::len(v2) != 2) {
                  throw std::invalid_argument("Tuples must have length 2");
              }
              double x1 = v1[0].cast<double>();
              double y1 = v1[1].cast<double>();
              double x2 = v2[0].cast<double>();
              double y2 = v2[1].cast<double>();
              return angle_between(x1, y1, x2, y2);
          },
          py::arg("v1"), py::arg("v2"),
          "Return angle (radians) between two (x,y) tuples.");
}
