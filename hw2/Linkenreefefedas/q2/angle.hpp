#pragma once
#include <cmath>
#include <stdexcept>

inline double angle_between(double ax, double ay, double bx, double by) {
    double na = std::hypot(ax, ay);
    double nb = std::hypot(bx, by);
    if (na == 0.0 || nb == 0.0) {
        throw std::invalid_argument("zero-length vector");
    }
    double dot   = ax*bx + ay*by;
    double cross = ax*by - ay*bx; // 2D "z" component
    return std::atan2(std::abs(cross), dot);
}
