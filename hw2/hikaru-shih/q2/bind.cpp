#include <pybind11/pybind11.h>
namespace py = pybind11;

double angle(double x1, double y1, double x2, double y2);

PYBIND11_MODULE(vector_angle, m) {
    m.doc() = "Calculate angle between two 2D vectors";
    m.def("angle", &angle, "Compute angle between vectors");
}
