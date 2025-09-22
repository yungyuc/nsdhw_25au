#include <cmath>
#include <limits>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// calculates angle of vector a and b
double angleOfTwoVectors(double a_x, double a_y, double b_x, double b_y)
{
    double dot_product = a_x * b_x + a_y * b_y;
    double magnitude_a = std::sqrt(a_x * a_x + a_y * a_y);
    double magnitude_b = std::sqrt(b_x * b_x + b_y * b_y);
    double denominator = magnitude_a * magnitude_b;

    // Prevent division by zero if either vector is a zero vector.
    // A small epsilon is used for floating-point comparison.
    if (denominator < std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Cannot calculate the angle of a zero vector.");

    double cos_theta = dot_product / denominator;
    if (cos_theta > 1.0)
    {
        cos_theta = 1.0;
    }
    if (cos_theta < -1.0)
    {
        cos_theta = -1.0;
    }

    return std::acos(cos_theta);
}

PYBIND11_MODULE(_vector, m)
{
    m.doc() = "pybind11 plugin";
    m.def("angle_of_two_vectors", &angleOfTwoVectors, "The function calculates angle of vector a and b, pass the variables as (vec1.x, vec1.y, vec2.x, vec2.y)");
}