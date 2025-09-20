#pragma once
#include <cmath>
#include <stdexcept>

inline double angle_between(double x1, double y1, double x2, double y2) {
    const double n1 = std::hypot(x1, y1);
    const double n2 = std::hypot(x2, y2);
    if (n1 == 0.0 || n2 == 0.0) {
        throw std::invalid_argument("Zero-length vector is not allowed");
    }
    const double dot   = x1 * x2 + y1 * y2;
    const double cross = x1 * y2 - y1 * x2;
    return std::atan2(std::abs(cross), dot);
}
